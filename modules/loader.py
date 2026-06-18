import os
from PyPDF2 import PdfReader  # pyre-ignore


class DocumentLoader:
    def __init__(self, dataset_path="docs"):
        self.dataset_path = dataset_path
        self.documents = {}

    def load_all_documents(self):
        if not os.path.exists(self.dataset_path):
            print(f"[ERROR] Folder '{self.dataset_path}' tidak ditemukan!")
            return self.documents

        for filename in sorted(os.listdir(self.dataset_path)):
            if filename.lower().endswith(".pdf"):
                filepath = os.path.join(self.dataset_path, filename)
                text = self._extract_text(filepath)
                if text.strip():
                    self.documents[filename] = text
                    print(f"  [OK] {filename}")
                else:
                    print(f"  [SKIP] {filename} (kosong)")

        print(f"  Total: {len(self.documents)} dokumen dimuat\n")
        return self.documents

    def _extract_text(self, filepath, max_pages=None, max_chars=None):
        text = ""

        try:
            reader = PdfReader(filepath)

            # ❗ cek apakah PDF punya halaman
            if not reader.pages:
                print(f"  [SKIP] {os.path.basename(filepath)} (tidak ada halaman)")
                return ""

            for i, page in enumerate(reader.pages):
                if max_pages is not None and i >= max_pages:
                    break

                try:
                    page_text = page.extract_text()

                    # ❗ kalau halaman tidak bisa dibaca
                    if not page_text or not page_text.strip():
                        continue

                    text += page_text + "\n"

                except Exception:
                    continue  # skip halaman error

        except Exception as e:
            print(f"  [SKIP] {os.path.basename(filepath)} (error: {e})")
            return ""

        # ❗ cek hasil akhir kosong
        if not text.strip():
            print(f"  [SKIP] {os.path.basename(filepath)} (tidak ada teks)")
            return ""

        if max_chars is not None:
            return text[:max_chars]
        return text
