# Create a tkinter window with a button that opens a file dialog
# to select a file and then prints the file path to the console.

from math import sqrt
import tkinter as tk
from tkinter import Frame, filedialog, messagebox, colorchooser
from PIL import Image, ImageTk, ImageColor
import functools

DEFAULT_THRESHOLD = .05
DEFAULT_MAX_COLOURS = 32


class FileDisplay(tk.Frame):
    def __init__(self, parent: Frame, root: Frame, button_text: str, default_text: str, command: callable):
        super().__init__(parent)
        self.parent = parent
        self.root = root
        self.button = tk.Button(self, text=button_text,
                                command=lambda: command(self))
        self.display = tk.Label(self, text=default_text)
        self.button.pack(side=tk.LEFT)
        self.display.pack(side=tk.LEFT)


class ImageFrame(tk.Frame):
    def __init__(self, parent: Frame, root: Frame, name: str, preview_size: int):
        super().__init__(parent, highlightbackground="black", highlightthickness=1)
        self.parent = parent
        self.root = root
        self.image = None
        self.label = tk.Label(self, text=name)
        self.label.pack(side=tk.TOP)
        self.canvas = tk.Canvas(
            self, bg="Grey", highlightbackground="black", highlightthickness=1)
        self.preview_size = preview_size
        self.canvas.configure(width=preview_size, height=preview_size)
        self.canvas.pack(side=tk.TOP)

    def set_image(self, image: Image.Image):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        width, height = image.size
        scale = min(canvas_width/width, canvas_height/height)
        self.input_resized = image.convert("RGB").resize(
            (int(width*scale), int(height*scale)), Image.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(self.input_resized)
        self.canvas.create_image(
            self.preview_size//2, self.preview_size//2, image=self.photo_image)


class InputOutputFrame(tk.Frame):
    def __init__(self, parent: Frame, root: Frame, preview_size: int):
        super().__init__(parent)
        self.parent = parent
        self.root = root
        self.input_frame = ImageFrame(self, root, 'Input', preview_size)
        self.output_frame = ImageFrame(self, root, 'Output', preview_size)
        self.input_frame.pack(side=tk.LEFT, anchor=tk.W)
        self.output_frame.pack(side=tk.RIGHT, anchor=tk.E)


class ColourBarGroup(tk.Frame):
    def __init__(self, parent: Frame, root: Frame):
        super().__init__(parent)
        self.parent = parent
        self.root = root
        self.input_bar = InputColourBar(self, root, 'Input')
        self.input_bar.pack(side=tk.LEFT, anchor=tk.W)
        self.output_bar = OutputColourBar(self, root, 'Output')
        self.output_bar.pack(side=tk.LEFT, anchor=tk.W)

    def set_both_colours(self, colours: list[tuple[int, int, int]]):
        self.input_bar.set_colours(colours)
        self.output_bar.set_colours(colours)

    def update_both_colours(self, colours: list[tuple[int, int, int]]):
        self.input_bar.update_colours(colours)
        self.output_bar.update_colours(colours)


class InputColourBar(tk.Frame):
    def __init__(self, parent, root, name: str):
        super().__init__(parent, highlightbackground="black", highlightthickness=1)
        self.parent = parent
        self.root = root
        self.box_size = root.preview_size//7
        self.label = tk.Label(self, text=name)
        self.label.pack(side=tk.TOP, expand=False)
        self.colour_container = tk.Frame(self, width=self.box_size,
                                         height=root.winfo_height())
        self.colour_container.pack(side=tk.TOP)
        self.colour_box_width = len("(255, 255, 255)")

    def create_button(self, colour: tuple[int, int, int], command=None):
        colour_box = tk.Button(self.colour_container)
        if command is not None:
            command = functools.partial(command, colour_box)
        self.set_button_colour(colour_box, colour)
        colour_box.configure(width=self.colour_box_width, height=0, command=command)
        colour_box.pack(side=tk.TOP, expand=True)

    def set_button_colour(self, button: tk.Button, colour: tuple[int, int, int]):
        colour_hex = f'#{colour[0]:02x}{colour[1]:02x}{colour[2]:02x}'
        avg_colour = sum(colour)//3
        text_colour = "black" if avg_colour > 128 else "white"
        button.configure(bg=colour_hex, text=str(colour), fg=text_colour)


    def cleanup(self):
        for child in self.colour_container.winfo_children():
            child.destroy()

    def set_colours(self, colours: list[tuple[int, int, int]]):
        self.cleanup()
        for colour in colours:
            self.create_button(colour)

    def update_colours(self, new_colours):
        # If the number of colours has changed add or remove boxes as necessary
        num_colours = len(new_colours)
        num_boxes = len(self.colour_container.winfo_children())
        if num_colours > num_boxes:
            for i in range(num_boxes, num_colours):
                colour = new_colours[i]
                self.create_button(colour)
        elif num_colours < num_boxes:
            for i in range(num_boxes, num_colours, -1):
                self.colour_container.winfo_children()[i-1].destroy()


class OutputColourBar(InputColourBar):
    def __init__(self, parent, root, name: str):
        super().__init__(parent, root, name)

    def set_colours(self, colours: list[tuple[int, int, int]]):
        self.cleanup()
        for colour in colours:
            self.create_button(colour, self.button_colour_set)

    def button_colour_set(self, button: tk.Button):
        current_colour = button["background"]
        rgb, _ = colorchooser.askcolor(
            title="Choose colour", initialcolor=current_colour)
        if rgb is None:
            return
        self.set_button_colour(button, rgb)
        self.root.application.calculate_output()


class ConfigBox(tk.Frame):
    def __init__(self, parent, root):
        super().__init__(parent, highlightbackground="black", highlightthickness=1)
        self.configure(width=root.preview_size//7 * 2, height=555)
        recalc_command = functools.partial(root.application.recalc_colours)
        self.parent = parent
        self.root: Application = root
        self.num_colours_label = tk.Label(self, text="Max colours:")
        self.num_colours_label.pack(side=tk.LEFT, expand=False)
        self.num_colours_var = tk.StringVar(value=str(DEFAULT_MAX_COLOURS))
        self.num_colours_var.trace("w", recalc_command)
        self.num_colours_value = tk.Spinbox(
            self, from_=1, to=32, increment=1, textvariable=self.num_colours_var)
        self.num_colours_value.pack(side=tk.LEFT, expand=False)

        self.threshold_label = tk.Label(
            self, text="Colour Threshold: (0.0 - 1.0)")
        self.threshold_label.pack(side=tk.LEFT, expand=False, padx=15)
        self.threshold_var = tk.StringVar(value=str(DEFAULT_THRESHOLD))
        self.threshold_var.trace("w", recalc_command)
        self.threshold_value = tk.Spinbox(
            self, from_=0, to=1, increment=0.01, textvariable=self.threshold_var)
        self.threshold_value.pack(side=tk.LEFT, expand=False)

        self.reset_button = tk.Button(
            self, text="Reset", command=self.root.application.reset_current_image)
        self.reset_button.pack(side=tk.LEFT, expand=False, padx=15)


class Application():
    def __init__(self):
        root = tk.Tk()
        root.title('Seb\'s Camo Recolourer')
        screen_size_ratio = 2/3
        window_width = int(root.winfo_screenwidth() * screen_size_ratio)
        # window_height = int(root.winfo_screenheight() * screen_size_ratio)
        # root.geometry(f'{window_width}x{window_height}')
        root.update()
        self.root = root
        self.input_image = None
        self.output_image = None
        root.application = self
        self.root.preview_size = window_width//2
        self.find_image_filetypes()
        self.init_ui()
        # self.refresh_ui()

    def find_image_filetypes(self):
        formats = [f"*{x}" for x in set(Image.registered_extensions().keys())]
        formats_str = ' '.join(formats)
        self.filetypes = (
            ('Image Files', formats_str),
            ('All files', '*.*')
        )

    def init_ui(self):
        root = self.root

        self.camo_load_frame = FileDisplay(
            root, root, 'Load Camo', 'No file selected', self.select_image_file)
        self.camo_load_frame.grid(row=0, column=0, sticky=tk.W)

        self.input_output_frame = InputOutputFrame(
            root, root, root.preview_size)
        self.input_output_frame.grid(row=1, column=0, columnspan=2)

        self.camo_save_frame = FileDisplay(
            root, root, 'Save Camo', 'No destination selected', self.save_image)
        self.camo_save_frame.grid(row=2, column=0, sticky=tk.W)

        self.config_box = ConfigBox(root, root)
        self.config_box.grid(row=0, column=1, sticky=tk.N + tk.W + tk.E + tk.S)

        self.colour_bar_group = ColourBarGroup(root, root)
        self.colour_bar_group.grid(row=1, column=2, rowspan=99, sticky=tk.N)

    def select_image_file(self, file_display: FileDisplay):
        filename = filedialog.askopenfilename(
            title='Open a file',
            initialdir='/',
            filetypes=self.filetypes)
        file_display.display['text'] = filename

        try:
            self.load_image(filename)
        except IOError:
            self.display['text'] = 'Invalid image file'
            return
        self.reset_current_image()
        if len(self.dominant_colours_brightness) == 0:
            messagebox.showerror(
                "Error", "No dominant colours found in image.\nTry decreasing the threshold or adding colours manually.")

    def save_image(self, file_display: FileDisplay):
        file_path = filedialog.asksaveasfilename(
            title='Save file',
            initialdir='/',
            filetypes=self.filetypes)
        if file_path == '':
            return
        try:
            self.output_image.save(file_path)
            file_display.display['text'] = f"Saved to {file_path}"
        except IOError:
            messagebox.showerror(
                "Error", "Could not save file. Please try again.")
            return

    def load_image(self, filename: str):
        input_canvas = self.input_output_frame.input_frame
        output_canvas = self.input_output_frame.output_frame
        self.input_image = Image.open(filename).convert(
            "P", palette=Image.ADAPTIVE)
        self.output_image = self.input_image.copy()
        input_canvas.set_image(self.input_image)
        output_canvas.set_image(self.input_image)
        # output_canvas.create_image(canvas_width//2, canvas_height//2, image=self.input_photoImage)

    def analyse_image(self, threshhold=None, max_colours=None):
        img = self.input_image.convert("RGB")
        # Percentage of pixels
        if threshhold is None:
            threshhold = float(self.config_box.threshold_var.get())

        num_pixels = img.width * img.height
        # Returns (num_pixels, colour)
        colours = img.getcolors(num_pixels)
        colours.sort(key=lambda x: x[0], reverse=True)
        if max_colours is None:
            max_colours = min(len(colours), int(
                self.config_box.num_colours_var.get()))
        colours = colours[:max_colours]
        threhshold_value = num_pixels * threshhold
        dominant_colours = [x[1]
                            for x in colours if x[0] > threhshold_value]
        # colours_dominant.sort(key=lambda x: rgb_to_hsv(*x[1])[2])
        dominant_colours_sorted = sorted(
            dominant_colours, key=lambda x: sqrt(sum(y**2 for y in x)))
        # Store index of dominant colours in original list
        self.dominant_colours_frequency = dominant_colours
        self.dominant_colours_brightness = dominant_colours_sorted
        # When sorted by brightness, the first colour is the darkest. Store index of highest frequency colour where 0 is the most frequent
        self.dominant_colour_indeces = [self.dominant_colours_frequency.index(
            x) for x in self.dominant_colours_brightness]

    def recalc_colours(self, *args):
        if self.input_image is None:
            return
        try:
            threshhold = float(self.config_box.threshold_var.get())
            max_colours = int(self.config_box.num_colours_var.get())
        except ValueError:
            return
        self.analyse_image(threshhold, max_colours)
        self.colour_bar_group.update_both_colours(
            self.dominant_colours_frequency)
        self.calculate_output()

    def reset_current_image(self):
        if self.input_image is None:
            return
        self.config_box.threshold_var.set(str(DEFAULT_THRESHOLD))
        self.config_box.num_colours_var.set(str(DEFAULT_MAX_COLOURS))
        self.analyse_image(threshhold=DEFAULT_THRESHOLD,
                           max_colours=DEFAULT_MAX_COLOURS)
        num_dominant_colours = len(self.dominant_colours_frequency)
        self.config_box.num_colours_var.set(str(num_dominant_colours))
        if num_dominant_colours == 0:
            messagebox.showerror(
                "Error", "No dominant colours found in image.\nTry increasing the threshold or adding colours manually.")
            return
        for colour_bar in [self.colour_bar_group.input_bar, self.colour_bar_group.output_bar]:
            colour_bar.set_colours(self.dominant_colours_frequency)
        self.calculate_output()

    def calculate_output(self):
        input_colours = self.colour_bar_group.input_bar.colour_container.winfo_children()
        output_colours = self.colour_bar_group.output_bar.colour_container.winfo_children()
        input_colours_unsorted = [ImageColor.getcolor(
            x["background"], "RGB") for x in input_colours]
        output_colours_unsorted = [ImageColor.getcolor(
            x["background"], "RGB") for x in output_colours]

        # Sort input colours by the frequency index
        input_colours = [input_colours_unsorted[i]
                         for i in self.dominant_colour_indeces]
        output_colours = [output_colours_unsorted[i]
                          for i in self.dominant_colour_indeces]

        conversion_table_rgb = list(zip(input_colours, output_colours))
        # print(input_colours)
        # print(output_colours)

        # Create new palette
        input_img = self.input_image.convert("P", palette=Image.ADAPTIVE)
        input_palette = input_img.getpalette()
        flag_palette_new = []
        for i in range(len(input_palette)//3):
            pixel = input_palette[i*3:i*3+3]
            pixel_new = convert_pixel_rgb(pixel, conversion_table_rgb)
            flag_palette_new.extend(pixel_new)
        output_img = input_img.copy()
        output_img.putpalette(flag_palette_new)
        output_img = output_img.convert("RGB")
        # flag_img.convert("RGB").show()
        # camo_img.convert("RGB").show()
        self.output_image = output_img
        self.input_output_frame.output_frame.set_image(output_img)

    def run(self):
        self.root.mainloop()


def convert_pixel_rgb(pixel: tuple[int, int, int], conversion_table_rgb: list[tuple[tuple[int, int, int], tuple[int, int, int]]]) -> tuple[int, int, int]:
    # If the magnitude is more than the magnitude of the last colour, use the last colour
    mag = sqrt(sum(x**2 for x in pixel))
    max_mag = sqrt(3*255**2)
    min_mag = 0
    if mag > max_mag or mag < min_mag:
        raise Exception("Pixel not in conversion table")
    smallest_mag = sqrt(sum(x**2 for x in conversion_table_rgb[0][0]))
    largest_mag = sqrt(sum(x**2 for x in conversion_table_rgb[-1][0]))

    # If the magnitude is less than the magnitude of the first colour, use the first colour
    if mag <= smallest_mag:
        return conversion_table_rgb[0][1]
    # If the magnitude is more than the magnitude of the last colour, use the last colour
    if mag >= largest_mag:
        return conversion_table_rgb[-1][1]

    # Otherwise, find the two colours to interpolate between
    for i, (next_source_colour, next_camo_colour) in enumerate(conversion_table_rgb):
        next_mag = sqrt(sum(x**2 for x in next_source_colour))
        if i == 0:
            prev_source_colour, prev_camo_colour = conversion_table_rgb[-1]
            prev_mag = sqrt(sum(x**2 for x in prev_source_colour)) - max_mag
        else:
            prev_source_colour, prev_camo_colour = conversion_table_rgb[i-1]
            prev_mag = sqrt(sum(x**2 for x in prev_source_colour))
        if prev_mag < mag <= next_mag:
            # Calculate the percentage of the way between the two colours
            amount_done = (mag - prev_mag) / (next_mag - prev_mag)
            # Interpolate between the two camo colours
            camo_colour = interpolate_colour_rgb(
                prev_camo_colour, next_camo_colour, amount_done)
            return camo_colour
    raise Exception("Pixel not in conversion table")


def interpolate_colour_rgb(colour1: tuple[int, int, int], colour2: tuple[int, int, int], amount_done: float) -> tuple[int, int, int]:
    # Interpolate each colour channel, accounting for wraparound
    colour = []
    for i in range(3):
        channel1 = colour1[i]
        channel2 = colour2[i]
        channel = channel1 + (channel2 - channel1) * amount_done
        channel %= 256
        colour.append(channel)
    return tuple(int(x) for x in colour)


def main():
    app = Application()
    app.run()


if __name__ == '__main__':
    main()
