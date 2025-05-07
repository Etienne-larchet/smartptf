import streamlit as st

from components.PageModels import PageModel, RenderWarning


class SolverPage(PageModel):
    def render(self):
        st.title("Solver")
        raise RenderWarning("Not yet implemented.")


if __name__ == "__main__":
    SolverPage().run()
