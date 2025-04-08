# main.py (entry point)
import sys
from PyQt5 import QtWidgets
from ui.main_window import MainWindow, show_splash_screen


def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    splash = show_splash_screen(main_window)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()