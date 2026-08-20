"""
main.py  --  Entry point.  Run with: python main.py
"""

from game import Game


def main():
    g = Game()
    try:
        g.run()
    except KeyboardInterrupt:
        pass   # Ctrl+C exits cleanly without traceback


if __name__ == "__main__":
    main()
