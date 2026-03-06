import tkinter as tk
from ui import OCRGUI


def main():
    root = tk.Tk()
    OCRGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()