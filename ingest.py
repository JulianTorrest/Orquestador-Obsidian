import os
import uuid
import db
import re
import graph_store
import json
import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

# --- Document Loaders ---
def load_pdf(file_path):
    """Extracts text from a PDF file."""
    import pypdf
    text = ""
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def load_docx(file_path):
    """Extracts text from a DOCX file."""
    import docx
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def load_md(file_path):
    """Extracts text from a Markdown file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def load_csv_excel(file_path):
    """
    Extracts text from CSV or Excel files using pandas, automatically
    detecting the header row and converting each row into a meaningful sentence.
    """
    import pandas as pd

    def find_header_row(df_preview):
        """Heuristic to find the header row by scoring the first few rows."""
        best_guess_index = -1
        max_score = -1

        for i, row in df_preview.head(20).iterrows():
            if row.isnull().all():
                continue

            # Score based on number of unique, non-numeric, non-empty values
            try:
                non_empty_cells = row.dropna()
                num_strings = sum(1 for x in non_empty_cells if isinstance(x, str))
                num_unique = non_empty_cells.nunique()
                
                # A good header has many unique string values
                score = num_strings + num_unique

                if score > max_score:
                    max_score = score
                    best_guess_index = i
            except Exception:
                continue
        
        return best_guess_index if best_guess_index != -1 else 0

    try:
        if file_path.endswith('.csv'):
            # Read first 20 rows without header to find the real one
            preview_df = pd.read_csv(file_path, header=None, nrows=20, sep=None, engine='python')
            header_row_index = find_header_row(preview_df)
            df = pd.read_csv(file_path, header=header_row_index, sep=None, engine='python')
        else:
            preview_df = pd.read_excel(file_path, header=None, nrows=20)
            header_row_index = find_header_row(preview_df)
            df = pd.read_excel(file_path, header=header_row_index)

        # Clean up column names (remove unnamed, etc.)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.columns = [str(col) for col in df.columns]

        # Convert each row to a descriptive string
        text_rows = []
        for _, row in df.iterrows():
            row_desc = ", ".join([f"{col}: '{row[col]}'" for col in df.columns if pd.notna(row[col])])
            if row_desc:
                text_rows.append(f"Registro con datos - {row_desc}.")
        return "\n".join(text_rows)

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def get_loader(file_extension):
    """Returns the appropriate loader function for a file extension."""
    loaders = {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".csv": load_csv_excel,
        ".xlsx": load_csv_excel,
        ".md": load_md,
    }
    return loaders.get(file_extension)

def ingest_documents(folder_path):
    """
    Walks through a directory, loads supported documents, splits them into chunks,
    and saves them to the database.
    """
    print("Starting document ingestion...")
    conn = db.get_connection()
    db.init_db(conn) # Ensure table exists
    
    # This splitter tries to keep paragraphs/sentences together.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    processed_files = 0

    try:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_name, file_extension = os.path.splitext(file)
                
                loader = get_loader(file_extension.lower())
                if not loader:
                    print(f"Skipping unsupported file type: {file}")
                    continue

                print(f"Processing {file_path}...")
                try:
                    text = loader(file_path)
                    if not text.strip():
                        print(f"No text extracted from {file}. Skipping.")
                        continue
                    
                    frontmatter = {}
                    
                    # Register the file as a node in the global knowledge graph
                    graph_store.update_graph_edges(file, [])
                    
                    # Extract links for GraphRAG
                    if file_extension.lower() == '.md':
                        # Extract YAML frontmatter
                        fm_match = re.match(r'^---\n(.*?)\n---', text, re.DOTALL)
                        if fm_match:
                            try:
                                frontmatter = yaml.safe_load(fm_match.group(1)) or {}
                                # Remove frontmatter from text so it doesn't pollute content
                                text = text[fm_match.end():]
                            except Exception as e:
                                print(f"Error parsing frontmatter in {file_name}: {e}")
                        
                        # Find all Obsidian style links [[Link]]
                        links = re.findall(r'\[\[(.*?)\]\]', text)
                        if links:
                            # Clean link names (remove aliases if present like [[Link|Alias]])
                            cleaned_links = [link.split('|')[0] for link in links]
                            graph_store.update_graph_edges(file, cleaned_links)
                    
                    if file_extension.lower() == '.md':
                        headers_to_split_on = [
                            ("#", "Header 1"),
                            ("##", "Header 2"),
                            ("###", "Header 3"),
                        ]
                        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
                        md_splits = md_splitter.split_text(text)
                        chunks = []
                        # Convert to string and append metadata
                        for doc in md_splits:
                            # Add frontmatter to metadata
                            doc.metadata.update(frontmatter)
                            header_info = " - ".join([f"{k}: {v}" for k, v in doc.metadata.items()])
                            chunk_text = f"{header_info}\n\n{doc.page_content}" if header_info else doc.page_content
                            chunks.extend(text_splitter.split_text(chunk_text))
                    else:
                        chunks = text_splitter.split_text(text)
                        
                    print(f"  > Split into {len(chunks)} chunks.")
                    
                    chunk_data = []
                    for i, chunk_content in enumerate(chunks):
                        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_path}_{i}"))
                        chunk_data.append((chunk_id, file, chunk_content, i))
                    
                    db.insert_chunks(conn, chunk_data)
                    print(f"  > Successfully saved {len(chunks)} chunks to DB.")
                    
                    # Save metadata (tags and frontmatter)
                    tags = frontmatter.get('tags', [])
                    if isinstance(tags, list):
                        tags_str = ", ".join(str(t) for t in tags)
                    else:
                        tags_str = str(tags)
                    fm_str = json.dumps(frontmatter, ensure_ascii=False) if frontmatter else "{}"
                    db.upsert_metadata(conn, file, tags_str, fm_str)
                    print(f"  > Metadata saved for {file}.")
                    
                    processed_files += 1

                except Exception as e:
                    print(f"An error occurred while processing {file_path}: {e}")
    except KeyboardInterrupt:
        print("\n\nProceso de ingesta interrumpido por el usuario. Saliendo de forma ordenada.")
    finally:
        conn.close()
        print(f"\nConnection closed. Processed {processed_files} files.")

if __name__ == '__main__':
    # --- Uso de rutas relativas para portabilidad ---
    # Esto construye la ruta a la carpeta de documentos de forma dinámica.
    # Asume que la carpeta 'RAG Pruebas' está al mismo nivel que la carpeta del proyecto 'mi_obsidian'.
    # __file__ es la ruta al script actual (ingest.py)
    # os.path.dirname(__file__) es la carpeta donde está el script (mi_obsidian)
    # os.path.join(..., '..') sube un nivel en la estructura de carpetas.
    project_dir = os.path.dirname(__file__)
    DOCS_FOLDER = os.path.abspath(os.path.join(project_dir, '..', 'RAG Pruebas'))
    
    ingest_documents(DOCS_FOLDER)
    print("Ingestion complete!")