try:
    from googletrans import Translator as GoogleTranslator
    GOOGLETRANS_AVAILABLE = True
except Exception:
    GOOGLETRANS_AVAILABLE = False

class TranslatorWrapper:
    def __init__(self):
        self.google = False
        if GOOGLETRANS_AVAILABLE:
            self.google = True
            self.gtr = GoogleTranslator()
            print("Googletrans available (online fallback).")

    def translate(self, text, src):
        if not text:
            return None
        try:
            if self.google:
                dest = 'hu' if src.startswith('en') else 'en'
                res = self.gtr.translate(text, src=src, dest=dest)
                return res.text
            return None
        except Exception as e:
            print("Translate error:", e)
            return None
