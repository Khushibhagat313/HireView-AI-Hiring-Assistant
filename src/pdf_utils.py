from pypdf import PdfReader

def extract_text(file_obj) -> str:
    try:
        reader = PdfReader(file_obj)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)
    except Exception as e:
        return ""
