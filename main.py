# Create a tkinter window with a button that opens a file dialog
# to select a file and then prints the file path to the console.

from math import sqrt
import tkinter as tk
from tkinter import Frame, filedialog, messagebox, colorchooser
from PIL import Image, ImageTk, ImageColor
from functools import partial, cache
from numpy import array, uint8, float64, reshape, append, argpartition, linalg, empty, array_equal, zeros
from sklearn.cluster import KMeans
from ast import literal_eval

DEFAULT_THRESHOLD = .025
DEFAULT_MAX_COLOURS = 1
VERSION = "v0.3.0"


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


class ColourBar(tk.Frame):
    def __init__(self, parent, root, name: str, command: callable):
        super().__init__(parent, highlightbackground="black", highlightthickness=1)
        self.parent = parent
        self.root = root
        self.box_size = root.preview_size//7
        self.label = tk.Label(self, text=name)
        self.label.pack(side=tk.TOP, expand=False)
        self.command = command
        self.colour_container = tk.Frame(self, width=self.box_size,
                                         height=root.winfo_height())
        self.colour_container.pack(side=tk.TOP)
        self.colour_box_width = len("(255, 255, 255)")
        self.colour_replacement_cache = {}

    def create_button(self, colour: tuple[int, int, int]):
        colour_box = tk.Button(self.colour_container)
        if self.command is not None:
            command = partial(self.command, colour_box, self)
        else:
            command = None
        self.set_button_colour(colour_box, colour)
        colour_box.configure(width=self.colour_box_width,
                             height=0, command=command)
        colour_box.pack(side=tk.TOP, expand=True)

    def set_button_colour(self, button: tk.Button, colour: tuple[int, int, int]):
        colour_hex = f'#{colour[0]:02x}{colour[1]:02x}{colour[2]:02x}'
        r, g, b = colour
        # Get avg colour using perceived brightness
        avg_colour = r*0.299 + g*0.587 + b*0.114
        text_colour = "black" if avg_colour > 128 else "white"
        button.configure(bg=colour_hex, text=str(colour), fg=text_colour)

    def cleanup(self):
        for child in self.colour_container.winfo_children():
            child.destroy()

    def set_colours(self, colours: list[tuple[int, int, int]]):
        self.cleanup()
        self.colour_replacement_cache = {}
        for colour in colours:
            self.create_button(colour)

    def update_colours(self, colours):
        # If the number of colours has changed add or remove boxes as necessary
        self.cleanup()
        for colour in colours:
            new_colour = self.colour_replacement_cache.get(colour, colour)
            self.create_button(new_colour)


class ColourBarGroup(tk.Frame):
    def __init__(self, parent: Frame, root: Frame):
        super().__init__(parent)
        self.parent = parent
        self.root = root
        self.input_bar = ColourBar(
            self, root, 'Input', self.bar_colour_button_change)
        self.input_bar.pack(side=tk.LEFT, anchor=tk.W)
        self.output_bar = ColourBar(
            self, root, 'Output', self.bar_colour_button_change)
        self.output_bar.pack(side=tk.LEFT, anchor=tk.W)

    def set_both_colours(self, colours: list[tuple[int, int, int]]):
        self.input_bar.set_colours(colours)
        self.output_bar.set_colours(colours)

    def update_both_colours(self, colours: list[tuple[int, int, int]]):
        self.input_bar.update_colours(colours)
        self.output_bar.update_colours(colours)

    def bar_colour_button_change(self, button: tk.Button, colour_bar: ColourBar):
        current_colour = ImageColor.getcolor(button["background"], "RGB")
        colour = colorchooser.askcolor(color=current_colour)[0]
        if colour is None or colour == current_colour:
            return
        colour_bar.colour_replacement_cache[current_colour] = colour
        colour_bar.set_button_colour(button, colour)
        self.root.application.calculate_output_3d()


class ConfigBox(tk.Frame):
    def __init__(self, parent, root):
        super().__init__(parent, highlightbackground="black", highlightthickness=1)
        self.configure(width=root.preview_size//7 * 2, height=555)
        recalc_command = partial(root.application.recalc_colours)
        self.parent = parent
        self.root: Application = root
        self.num_colours_label = tk.Label(self, text="Number of colours:")
        self.num_colours_label.pack(side=tk.LEFT, expand=False)
        self.num_colours_var = tk.StringVar(value=str(DEFAULT_MAX_COLOURS))
        self.num_colours_var.trace("w", recalc_command)
        self.num_colours_value = tk.Spinbox(
            self, from_=1, to=32, increment=1, textvariable=self.num_colours_var)
        self.num_colours_value.pack(side=tk.LEFT, expand=False)

        # self.threshold_label = tk.Label(
        # self, text="Colour Threshold: (0.0 - 1.0)")
        # self.threshold_label.pack(side=tk.LEFT, expand=False, padx=15)
        self.threshold_var = tk.StringVar(value=str(DEFAULT_THRESHOLD))
        # self.threshold_var.trace("w", recalc_command)
        # self.threshold_value = tk.Spinbox(
        #     self, from_=0, to=1, increment=0.01, textvariable=self.threshold_var)
        # self.threshold_value.pack(side=tk.LEFT, expand=False)

        self.reset_button = tk.Button(
            self, text="Reset", command=self.root.application.reset_current_image)
        self.reset_button.pack(side=tk.LEFT, expand=False, padx=15)


class Application():
    def __init__(self):
        root = tk.Tk()
        root.title(f'Seb\'s Camo Recolourer {VERSION}')
        screen_size_ratio = 2/3
        window_width = int(root.winfo_screenwidth() * screen_size_ratio)
        # window_height = int(root.winfo_screenheight() * screen_size_ratio)
        # root.geometry(f'{window_width}x{window_height}')
        root.update()
        self.suspend_recalc = True
        self.root = root
        self.input_image = None
        self.output_image = None
        root.application = self
        self.root.preview_size = window_width//2
        self.find_image_filetypes()
        self.init_ui()
        self.suspend_recalc = False
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
        self.camo_load_frame.grid(row=0, column=0, columnspan=3, sticky=tk.W)

        self.config_box = ConfigBox(root, root)
        self.config_box.grid(row=0, column=3, sticky=tk.W)

        self.colour_bar_group = ColourBarGroup(root, root)
        self.colour_bar_group.grid(row=1, column=0, columnspan=2, sticky=tk.N)

        self.input_output_frame = InputOutputFrame(
            root, root, root.preview_size)
        self.input_output_frame.grid(row=1, column=2, columnspan=2)

        self.camo_save_frame = FileDisplay(
            root, root, 'Save Camo', 'No destination selected', self.save_image)
        self.camo_save_frame.grid(row=2, column=0, sticky=tk.W)

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

    def load_image(self, file_path: str):
        input_canvas = self.input_output_frame.input_frame
        output_canvas = self.input_output_frame.output_frame
        if file_path == '':
            return
        self.file_path = file_path
        self.input_image = Image.open(file_path).convert(
            "RGB", palette=Image.ADAPTIVE)
        self.output_image = self.input_image.copy()
        input_canvas.set_image(self.input_image)
        output_canvas.set_image(self.input_image)

    def estimate_num_dominant_colours(self, img: Image.Image, threshhold=None):
        # Percentage of pixels
        if threshhold is None:
            threshhold = float(self.config_box.threshold_var.get())
        num_pixels = img.width * img.height
        # Returns (num_pixels, colour)
        colours = img.getcolors(num_pixels)
        threhshold_value = num_pixels * threshhold
        dominant_colours = [x[1]
                            for x in colours if x[0] > threhshold_value]
        return max(len(dominant_colours), 1)

    def k_cluster_analysis(self, threshold=None, num_colours=None):
        if self.input_image is None:
            return
        if num_colours is None:
            num_colours = self.estimate_num_dominant_colours(
                self.input_image, threshold)

        dominant_colours_frequency, dominant_colours_brightness = k_cluster_main(
            num_colours, self.file_path)
        # Store index of dominant colours in original list
        self.dominant_colours_frequency = dominant_colours_frequency
        self.dominant_colours_brightness = dominant_colours_brightness
        # When sorted by brightness, the first colour is the darkest. Store index of highest frequency colour where 0 is the most frequent
        self.dominant_colour_indeces = [self.dominant_colours_frequency.index(
            x) for x in self.dominant_colours_brightness]

    def recalc_colours(self, *_):
        if self.input_image is None or self.suspend_recalc:
            return
        try:
            threshhold = float(self.config_box.threshold_var.get())
            max_colours = int(self.config_box.num_colours_var.get())
        except ValueError:
            return
        self.k_cluster_analysis(threshhold, max_colours)
        self.colour_bar_group.update_both_colours(
            self.dominant_colours_brightness)
        self.calculate_output()

    def reset_current_image(self):
        if self.input_image is None:
            return
        self.suspend_recalc = True
        self.config_box.threshold_var.set(str(DEFAULT_THRESHOLD))
        self.config_box.num_colours_var.set(str(DEFAULT_MAX_COLOURS))
        self.k_cluster_analysis(DEFAULT_THRESHOLD, None)
        num_dominant_colours = len(self.dominant_colours_frequency)
        self.config_box.num_colours_var.set(str(num_dominant_colours))
        self.suspend_recalc = False
        if num_dominant_colours == 0:
            messagebox.showerror(
                "Error", "No dominant colours found in image.\nTry decreasing the threshold then increasing the number of colours.")
            return
        self.colour_bar_group.set_both_colours(
            self.dominant_colours_brightness)
        self.calculate_output()


    def calculate_output(self):
        print("Calculating output in 3d")
        if self.input_image is None:
            return
        input_colour_boxes = self.colour_bar_group.input_bar.colour_container.winfo_children()
        output_colour_boxes = self.colour_bar_group.output_bar.colour_container.winfo_children()
        if max(len(input_colour_boxes), len(output_colour_boxes)) == 0:
            return
        input_colours = tuple(literal_eval(x["text"])
                              for x in input_colour_boxes)
        output_colours = tuple(literal_eval(
            x["text"]) for x in output_colour_boxes)

        input_img = self.input_image.convert("P", palette=Image.ADAPTIVE)
        # Bad hash, but it works
        input_palette_flat = tuple(input_img.getpalette())
        output_palette = calc_new_palette(
            input_colours, output_colours, input_palette_flat)
        # Flatten the output palette
        output_img = input_img.copy()
        output_palette = list(output_palette.astype(uint8).flatten())
        output_img.putpalette(output_palette)
        output_img = output_img.convert("RGB")
        # flag_img.convert("RGB").show()
        # camo_img.convert("RGB").show()
        self.output_image = output_img
        self.input_output_frame.output_frame.set_image(output_img)

    def run(self):
        self.root.mainloop()


@cache
def calc_new_palette(input_colours, output_colours, input_palette_flat: str):
    # Palette provied as str so it can be cached
    input_palette = array((input_palette_flat), dtype=uint8).reshape(-1, 3)
    input_colours = array(input_colours, dtype=uint8)
    output_colours = array(output_colours, dtype=uint8)
    output_palette = empty((0, 3), dtype=uint8)
    num_weight_pixels = min(len(input_colours), 2)
    for input_pixel in input_palette:
        if num_weight_pixels < 2:
            output_palette = append(
                output_palette, output_colours[0].reshape(1, 3), axis=0)
            continue
        # Treat pixel as 3d coordinate, find closest 4 input colours in input_colours
        # Find the 3 closest colours in input_colours
        distances = linalg.norm(abs(input_colours.astype(
            float64) - input_pixel.astype(float64)), axis=1)
        # Get closest 3 colours
        closest_indeces = argpartition(
            distances, num_weight_pixels-1)[:num_weight_pixels]

        # If the closest colour is the same as the input colour, add it to the output palette from the same index
        if array_equal(array([152, 128, 102], dtype=uint8), input_pixel):
            print("Found 152, 128, 102")
        closest_pixel = input_colours[closest_indeces[0]]
        if array_equal(closest_pixel, input_pixel):
            output_palette = append(
                output_palette, output_colours[closest_indeces[0]].reshape(1, 3), axis=0)
            continue
        # Weight the output colour based on the distance to the closest colour
        output_pixel = zeros(3, dtype=uint8)
        closest_distances = distances[closest_indeces]**2
        total_distance = sum(closest_distances)
        weights_unnormal = 1 - closest_distances/total_distance
        weights = weights_unnormal/sum(weights_unnormal)
        output_pixel = sum(
            output_colours[closest_indeces] * weights.reshape(-1, 1))
        # Normalise the output pixel
        output_palette = append(
            output_palette, output_pixel.reshape(1, 3), axis=0)
    return output_palette


@cache
def k_cluster_main(num_colours: int, path: str):
    target_px = 1024**2
    img = Image.open(path).convert("P", palette=Image.ADAPTIVE)

    num_px = img.width * img.height
    reduction_factor = max(sqrt(num_px/target_px), 1)
    if reduction_factor > 1:
        new_x = int(img.width/reduction_factor)
        new_y = int(img.height/reduction_factor)
        img = img.resize((new_x, new_y), Image.NEAREST)

    img_array = array(img.convert("RGB", palette=Image.ADAPTIVE))
    img2D = img_array.reshape(-1, 3)
    # Apply KMeans clustering
    n_init = 20
    kmeans_model = KMeans(n_clusters=num_colours,
                          n_init=n_init, random_state=0, tol=1e-10)
    cluster_labels = kmeans_model.fit_predict(img2D)

    # Get the cluster centres
    cluster_centres = kmeans_model.cluster_centers_
    rgb_colours = cluster_centres.astype(int)
    quantized_img = Image.fromarray(
        reshape(rgb_colours[cluster_labels], (img_array.shape)).astype(uint8))

    # Get the frequency of each cluster
    num_pixels = img.width * img.height
    dominant_colours = quantized_img.getcolors(num_pixels)
    dominant_colours.sort(key=lambda x: x[0], reverse=True)
    dominant_colours_frequency = [x[1] for x in dominant_colours]
    dominant_colours_brightness = sorted(
        dominant_colours_frequency, key=lambda x: sqrt(sum(y**2 for y in x)))
    return dominant_colours_frequency, dominant_colours_brightness


def main():
    app = Application()
    app.run()


if __name__ == '__main__':
    main()
