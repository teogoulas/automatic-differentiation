from kivy.app import App
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.stencilview import StencilView
from kivy.graphics import Color, Ellipse, Rectangle


class MyPaintWidget(StencilView):
    def on_touch_down(self, touch):
        color = (1, 1, 1)
        with self.canvas:
            Color(*color)
            d = 30.
            Ellipse(pos=(touch.x - d / 2., touch.y - d / 2.), size=(d, d))

    def on_touch_move(self, touch):
        color = (1, 1, 1)
        with self.canvas:
            Color(*color)
            d = 30.
            Ellipse(pos=(touch.x - d / 2., touch.y - d / 2.), size=(d, d))


class MyPaintApp(App):
    def build(self):
        parent = Widget()
        self.painter = MyPaintWidget(size=[Window.size[0], Window.size[1]])
        with self.painter.canvas:
            Color(0, 0, 0, 255)
            Rectangle(pos=self.painter.pos, size=self.painter.size)
        ssbtn = Button(text='Save')
        ssbtn.bind(on_release=self.save_screenshot)
        parent.add_widget(self.painter)
        parent.add_widget(ssbtn)
        return parent

    def save_screenshot(self, obj):
        self.painter.export_to_png("drawing/digit.png")
        App.get_running_app().stop()


def open_window():
    MyPaintApp().run()
