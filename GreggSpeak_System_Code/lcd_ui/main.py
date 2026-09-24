import sys
from pathlib import Path


LCD_DIR = Path(__file__).resolve().parent
if str(LCD_DIR) not in sys.path:
    sys.path.insert(1, str(LCD_DIR))

from lcd_ui.app import App

if __name__ == "__main__":
    app = App()
    app.mainloop()
