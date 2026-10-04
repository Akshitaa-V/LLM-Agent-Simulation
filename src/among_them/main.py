import sys
from pathlib import Path

import streamlit as st

# Allow running `streamlit run src/among_them/main.py` without Poetry.
if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from among_them.game.game_engine import GameEngine
from among_them.gui_handler import GUIHandler

# To run this script, you need to
# `poetry install`
# and then run the following command:
# `poetry run main`


def main():
    st.set_page_config(page_title="Among Them", layout="wide")
    gui_handler = GUIHandler()
    
    # Inject JavaScript to remove the footer
    #js = """
    #<script>
    #    function getTopWindow(currentWindow) {
    #        if (currentWindow.parent === currentWindow) return currentWindow;
    #        return getTopWindow(currentWindow.parent);
    #    }
    #
    #    document.addEventListener('DOMContentLoaded', () => {
    #        try {
    #            const topWindow = getTopWindow(window);
    #            const divs = topWindow.document.getElementsByTagName('div');
    #            Array.from(divs).forEach(div => {
    #                if (div.className?.includes('_profileContainer')) {
    #                    div.remove();
    #                }
    #            });
    #        } catch (err) {}
    #    });
    #</script>
    #"""
    #st.components.v1.html(js, height=0)
    
    game_engine = GameEngine()

    game_engine.load_game()
    game_engine.state.DEBUG = True

    gui_handler.display_gui(game_engine)


if __name__ == "__main__":
    main()
