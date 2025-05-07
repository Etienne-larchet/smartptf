from abc import ABC, abstractmethod

import streamlit as st


class RenderWarning(Exception):
    pass


class RenderInfo(Exception):
    pass


class StreamModel(ABC):
    def run(self):
        try:
            self.render_before()
            self.render()
            self.render_after()
        except RenderWarning as e:
            st.warning(e)
        except RenderInfo as e:
            st.info(e)
        except Exception as e:
            st.error(e)

    @abstractmethod
    def render_before(self):
        pass

    @abstractmethod
    def render_after(self):
        pass

    @abstractmethod
    def render(self): ...


class PageModel(StreamModel):
    def render_before(self):
        pass

    def render_after(self):
        pass
