# Create a tkinter window with a button that opens a file dialog
# to select a file and then prints the file path to the console.

from math import sqrt
from pathlib import Path
import tkinter as tk
from tkinter import Frame, filedialog, messagebox, colorchooser
from tkinter import ttk
from PIL import Image, ImageTk, ImageColor, ImageDraw
from functools import partial, cache
from numpy import array, uint8, float64, reshape, append, argpartition, linalg, empty, array_equal, zeros
from sklearn.cluster import KMeans
from scipy.stats import mode
from ast import literal_eval

DEFAULT_THRESHOLD = .025
DEFAULT_MAX_COLOURS = 1
VERSION = "v0.5.1"


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
        self.last_file_path: str = None


class ColourTransposer(tk.Frame):
    def __init__(self, parent: Frame, root: Frame, button_text: str, default_text: str, command: callable):
        super().__init__(parent)
        self.parent = parent
        self.root = root
        self.file_path = None
        self.button = tk.Button(self, text=button_text,
                                command=lambda: command(self))

        self.instruction = tk.Label(self, text="Match transpose colours by:")

        self.transpose_modes = ["Brightness", "Frequency"]
        self.transpose_mode = tk.StringVar(value=self.transpose_modes[0])
        self.transpose_mode.trace("w", self.transpose_mode_changed)
        self.transpose_mode_dropdown = ttk.Combobox(
            self, textvariable=self.transpose_mode, values=self.transpose_modes)

        self.display = tk.Label(self, text=default_text)

        self.button.pack(side=tk.LEFT)
        self.instruction.pack(side=tk.LEFT, padx=35)
        self.transpose_mode_dropdown.pack(side=tk.LEFT)
        self.display.pack(side=tk.LEFT, padx=15)

    def transpose_mode_changed(self, *_):
        if self.transpose_mode.get() not in self.transpose_modes:
            self.transpose_mode.set(self.transpose_modes[0])
            return
        if self.file_path is None:
            return
        self.root.application.transpose_colours(self)


class ImageFrame(tk.Frame):
    def __init__(self, parent: Frame, root: Frame, name: str, preview_size: int):
        super().__init__(parent, highlightbackground="black", highlightthickness=1)
        self.parent = parent
        self.root = root
        self.image = None
        self.label = tk.Label(self, text=name)
        self.label.pack(side=tk.TOP)
        self.image_canvas = tk.Canvas(
            self, bg="Grey", highlightbackground="black", highlightthickness=1)
        self.preview_size = preview_size
        self.image_canvas.configure(width=preview_size, height=preview_size)
        self.image_canvas.pack(side=tk.TOP)
        self.histogram_scale = self.get_histogram_scale()
        self.histogram_canvas = tk.Canvas(
            self, bg="Grey", highlightbackground="black", highlightthickness=1)
        self.histogram_canvas.configure(
            width=preview_size, height=256*self.histogram_scale)
        self.histogram_canvas.pack(side=tk.TOP)

    def set_image(self, image: Image.Image):
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        width, height = image.size
        scale = min(canvas_width/width, canvas_height/height)
        self.input_resized = image.convert("RGB").resize(
            (int(width*scale), int(height*scale)), Image.LANCZOS)
        self.image_photo_image = ImageTk.PhotoImage(self.input_resized)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(
            self.preview_size//2, self.preview_size//2, image=self.image_photo_image)
        self.draw_histogram(image)

    def draw_histogram(self, Image: Image.Image):
        width, height = self.histogram_canvas.winfo_width(
        ), self.histogram_canvas.winfo_height()
        histogram_image = create_histogram(Image, (width, height))
        self.histogram_photo_image = ImageTk.PhotoImage(histogram_image)
        self.histogram_canvas.delete("all")
        self.histogram_canvas.create_image(
            0, 0, image=self.histogram_photo_image, anchor=tk.NW)

    def get_histogram_scale(self):
        canvas_width = self.image_canvas.winfo_width()
        histogram_width = canvas_width/3
        return max(1, int(histogram_width/256))


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
        self.colour_replacement_cache = {}

    def create_button(self, colour: tuple[int, int, int]):
        colour_box = tk.Button(self.colour_container)
        command = partial(self.bar_colour_button_change, colour_box)
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
        self.cleanup()
        for colour in colours:
            new_colour = self.colour_replacement_cache.get(colour, colour)
            self.create_button(new_colour)

    def bar_colour_button_change(self, button: tk.Button):
        current_colour = ImageColor.getcolor(button["background"], "RGB")
        colour = colorchooser.askcolor(color=current_colour)[0]
        if colour is None or colour == current_colour:
            return
        self.colour_replacement_cache[current_colour] = colour
        self.set_button_colour(button, colour)
        self.root.application.calculate_output()


class ColourBarGroup(tk.Frame):
    def __init__(self, parent: Frame, root: Frame):
        super().__init__(parent)
        self.parent = parent
        self.root = root
        self.input_bar = ColourBar(
            self, root, 'Input')
        self.input_bar.pack(side=tk.LEFT, anchor=tk.W)
        self.output_bar = ColourBar(
            self, root, 'Output')
        self.output_bar.pack(side=tk.LEFT, anchor=tk.W)

    def set_both_colours(self, colours: list[tuple[int, int, int]]):
        self.input_bar.set_colours(colours)
        self.output_bar.set_colours(colours)

    def update_both_colours(self, colours: list[tuple[int, int, int]]):
        self.input_bar.update_colours(colours)
        self.output_bar.update_colours(colours)


class ConfigBox(tk.Frame):
    def __init__(self, parent, root):
        super().__init__(parent, highlightbackground="black", highlightthickness=1)
        self.configure(width=root.preview_size//7 * 2, height=555)
        recalc_command = partial(root.application.recalc_input)
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

        self.colour_transposer = ColourTransposer(
            root, root, 'Transpose Colours From Camo', "no file selected", self.transpose_colours_dialog)
        self.colour_transposer.grid(row=1, column=0, columnspan=3, sticky=tk.W)

        self.config_box = ConfigBox(root, root)
        self.config_box.grid(row=0, column=3, sticky=tk.W)

        self.colour_bar_group = ColourBarGroup(root, root)
        self.colour_bar_group.grid(row=2, column=0, columnspan=2, sticky=tk.N)

        self.input_output_frame = InputOutputFrame(
            root, root, root.preview_size)
        self.input_output_frame.grid(row=2, column=2, columnspan=2)

        self.camo_save_frame = FileDisplay(
            root, root, 'Save Camo', 'No destination selected', self.save_image)
        self.camo_save_frame.grid(row=3, column=0, sticky=tk.W)

        self.camo_save_as_layers_frame = FileDisplay(
            root, root, 'Save Camo as Layers', 'No destination selected', self.save_as_layers)
        self.camo_save_as_layers_frame.grid(row=4, column=0, sticky=tk.W)

    def transpose_colours_dialog(self, transposer: ColourTransposer):
        if self.input_image is None:
            transposer.display['text'] = "No input image selected!"
            return

        previous_file_path = transposer.file_path or '/'
        file_path = filedialog.askopenfilename(
            title='Open a file',
            initialdir=previous_file_path,
            filetypes=self.filetypes)
        if file_path == '':
            return
        transposer.file_path = file_path
        transposer.display['text'] = file_path

        self.transpose_colours(transposer)

    def transpose_colours(self, transposer):
        num_colours = int(self.config_box.num_colours_var.get())
        try:
            colours_frequency, colours_brightness = k_cluster_main(
                num_colours, transposer.file_path)
            if len(colours_frequency) < num_colours:
                self.config_box.num_colours_var.set(
                    str(len(colours_frequency)))
                # self.recalc_input()

            if transposer.transpose_mode.get() == transposer.transpose_modes[0]:
                new_palette = colours_brightness
            else:
                new_palette = [colours_frequency[x]
                               for x in self.dominant_colour_frequency_indeces]

            self.colour_bar_group.output_bar.update_colours(new_palette)
            self.calculate_output()
        except (IOError, AttributeError) as e:
            transposer.display['text'] = e
            return

    def select_image_file(self, file_display: FileDisplay):
        initial_dir = file_display.last_file_path or '/'
        file_path = filedialog.askopenfilename(
            title='Open a file',
            initialdir=initial_dir,
            filetypes=self.filetypes)
        file_display.display['text'] = file_path

        if file_path == '':
            return

        file_display.last_file_path = file_path
        try:
            self.load_image(file_path)
        except IOError:
            self.display['text'] = 'Invalid image file'
            return
        self.reset_current_image()

    def save_image(self, file_display: FileDisplay):
        last_file_path = file_display.last_file_path
        file_path = filedialog.asksaveasfilename(
            title='Save file',
            initialdir=last_file_path,
            filetypes=self.filetypes)
        if file_path == '':
            return
        file_display.last_file_path = str(last_file_path)
        # If file has invalid extension, add .png
        file_path = Path(file_path)
        if file_path.suffix not in Image.registered_extensions():
            file_path = file_path.with_suffix('.png')
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
        # When sorted by brightness, the first colour is the darkest. Store index of highest frequency colour in the brightness list
        self.dominant_colour_frequency_indeces = [
            dominant_colours_frequency.index(x) for x in dominant_colours_brightness]

    def recalc_input(self, *_):
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
        self.colour_transposer.file_path = None
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
        output_img.putpalette(output_palette)
        output_img = output_img.convert("RGB")
        # flag_img.convert("RGB").show()
        # camo_img.convert("RGB").show()
        self.output_image = output_img
        self.input_output_frame.output_frame.set_image(output_img)

    def save_as_layers(self, file_display: FileDisplay):
        initial_dir = file_display.last_file_path or '/'
        path = filedialog.asksaveasfilename(
            title='Save file',
            initialdir=initial_dir,
            filetypes=self.filetypes)
        if path == '':
            return
        file_display.last_file_path = path
        # Forcibly add .png extension
        path = Path(path).with_suffix('.png')
        try:
            seperate_layers(self.output_image,
                            self.dominant_colours_brightness, path)
            file_display.display['text'] = f"Saved to {path}"
        except IOError:
            messagebox.showerror(
                "Error", "Could not save file. Please try again.")
            return

    def run(self):
        self.root.mainloop()


def seperate_layers(img: Image.Image, input_colours: list[tuple[int, int, int]], path: Path):
    # Palettise image
    if len(input_colours) < 2:
        return
    input_colours = array(input_colours, dtype=uint8)
    img = img.convert("P", palette=Image.ADAPTIVE)
    # Get palette as 2d array
    palette = array(img.getpalette(), dtype=uint8).reshape(-1, 3)
    palette_zeros = zeros(palette.shape, dtype=uint8)
    # For each colour in the palette, find the closest 2 colours in the input palette.
    # Use the distance to build an alpha mask based on the distance to the closest colour and the distance to the second closest colour
    output_palettes = [palette_zeros.copy() for _ in range(len(input_colours))]
    for i, pixel in enumerate(palette):
        # Find the 2 closest colours in input_colours
        distances = linalg.norm(abs(input_colours.astype(
            float64) - pixel.astype(float64)), axis=1)
        # Get closest 2 colours
        closest_indeces = argpartition(
            distances, 1)[:2]
        # If the closest colour is the same as the input colour, add it to the output palette from the same index
        closest_index = closest_indeces[0]
        closest_pixel = input_colours[closest_index]
        closest_palette = output_palettes[closest_index]
        if array_equal(closest_pixel, pixel):
            closest_palette[i] = [255, 255, 255]
            continue
        # If neither of the closest colours are the same as the input colour, set to black
        second_closest_index = closest_indeces[1]

        distance_to_closest = distances[closest_index]
        distance_to_second_closest = distances[second_closest_index]
        total_distance = distance_to_closest + distance_to_second_closest

        # # Weight the output colour based on the distance to the closest colour
        if distance_to_closest <= total_distance*0.5:
            closest_palette[i] = [255, 255, 255]
        # threshhold = 0.25
        # threshhold_distance = total_distance*threshhold
        # if distance_to_closest <= threshhold_distance:
        #     closest_palette[i] = [255, 255, 255]
        # else:
        #     distance_remaining = total_distance - threshhold_distance
        #     weight = 1-(distance_to_closest/distance_remaining)
        #     brightness = int(255*weight)
        #     closest_palette[i] = [brightness, brightness, brightness]

    # Save each image
    for i, output_palette in enumerate(output_palettes):
        output_path = path.with_name(f"{path.stem}_{i}{path.suffix}")
        output_alpha = img.copy()
        output_alpha.putpalette(output_palette.flatten())
        output_alpha = output_alpha.convert("L")
        # output_alpha.show()
        output_img = img.copy().convert("RGB")
        output_img.putalpha(output_alpha)
        # output_img.show()
        output_img.save(output_path)


@cache
def calc_new_palette(input_colours, output_colours, input_palette_flat):
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
        # Treat pixel as 3d coordinate, find closest 2 input colours in input_colours
        # Find the 2 closest colours in input_colours
        distances = linalg.norm(abs(input_colours.astype(
            float64) - input_pixel.astype(float64)), axis=1)
        # Get closest 2 colours
        closest_indeces = argpartition(
            distances, num_weight_pixels-1)[:num_weight_pixels]

        # If the closest colour is the same as the input colour, add it to the output palette from the same index
        closest_pixel = input_colours[closest_indeces[0]]
        if array_equal(closest_pixel, input_pixel):
            output_palette = append(
                output_palette, output_colours[closest_indeces[0]].reshape(1, 3), axis=0)
            continue
        # Weight the output colour based on the distance to the closest colour
        output_pixel = zeros(3, dtype=uint8)
        closest_distances = distances[closest_indeces]
        total_distance = sum(closest_distances)
        weights_unnormal = 1 - closest_distances/total_distance
        weights = weights_unnormal/sum(weights_unnormal)
        output_pixel = sum(
            output_colours[closest_indeces] * weights.reshape(-1, 1))
        # Normalise the output pixel
        output_palette = append(
            output_palette, output_pixel.reshape(1, 3), axis=0)
    return list(output_palette.astype(uint8).flatten())


@cache
def k_cluster_main(num_colours: int, path: str):
    img = Image.open(path).convert("P", palette=Image.ADAPTIVE)
    num_px = img.width * img.height
    target_px = 256**2
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
                          n_init=n_init, random_state=0)
    cluster_labels = kmeans_model.fit_predict(img2D)

    # Get the cluster centres
    # cluster_centres = kmeans_model.cluster_centers_
    cluster_modes = empty((num_colours, 3), dtype=float64)
    for i in range(num_colours):
        cluster = img2D[cluster_labels == i]
        cluster_mode = mode(cluster, axis=0).mode
        cluster_modes[i] = cluster_mode
    rgb_colours = cluster_modes.astype(int)
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


def create_histogram(img: Image.Image, image_size: tuple[int, int]) -> Image.Image:
    histogram_length = 256
    histogram_size = (histogram_length, histogram_length)
    img = img.convert("RGB")
    bands = tuple(x.histogram() for x in img.split())
    num_pixels = max(img.histogram())
    band_images = []
    alpha = Image.new("L", histogram_size, "black")
    alpha_draw = ImageDraw.Draw(alpha)
    for band in bands:
        band_image = Image.new("L", histogram_size, "black")
        draw = ImageDraw.Draw(band_image)
        for x_val, y_val in enumerate(band):
            y_val = int(y_val/num_pixels*histogram_length)
            x = x_val
            y = y_val
            upper_left = (x, histogram_length-y)
            lower_right = (x, histogram_length)
            draw.rectangle((upper_left, lower_right), fill=255)
            alpha_draw.rectangle((upper_left, lower_right), fill=255)
        band_images.append(band_image)
        # band_image.show()
    histogram = Image.merge("RGB", band_images)
    histogram.putalpha(alpha)
    lines_overlay = Image.new("RGBA", histogram_size, (128, 128, 128, 255))
    lines_draw = ImageDraw.Draw(lines_overlay)
    for i in range(0, histogram_length, 32):
        lines_draw.line((i, 0, i, histogram_length), fill=(0, 0, 0, 255))
        lines_draw.line((0, i, histogram_length, i), fill=(0, 0, 0, 255))
    histogram = Image.alpha_composite(lines_overlay, histogram)
    # histogram.show()
    histogram = histogram.resize(image_size, Image.NEAREST)
    # histogram.show()
    return histogram


def main():
    app = Application()
    app.run()


if __name__ == '__main__':
    main()
