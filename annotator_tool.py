import argparse
import json
import os
import sys
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, ttk

try:
    from PIL import Image, ImageDraw, ImageOps, ImageTk
except ImportError:
    print("ERROR: Pillow is required. Install with: pip install Pillow", file=sys.stderr)
    sys.exit(1)

CHOICE_LETTERS = ["A", "B", "C", "D", "E", "F"]
IMAGE_CELL_SIZE = 400  # pixels for each image slot (square)

# Color palette
BG          = "#F7F8FA"   # main window background
HEADER_BG   = "#2C3E50"   # header bar (dark navy)
HEADER_FG   = "#FFFFFF"   # header text
IMG_BG      = "#E8EAED"   # image panel background
PANEL_BG    = "#FFFFFF"   # fields panel background
ENTRY_BG    = "#FFFFFF"   # text entry backgrounds
ENTRY_FG    = "#1A1A1A"   # text entry foreground
LABEL_FG    = "#2C3E50"   # section label text
FLIP_FG     = "#E67E22"   # orange for [FLIP ON] badge
BTN_NAV_BG  = "#2B9AE4"   # Prev / Next buttons (blue)
BTN_NAV_FG  = "#FFFFFF"
BTN_ADD_BG  = "#27AE60"   # Add button (green)
BTN_ADD_FG  = "#FFFFFF"
STATUS_BG   = "#ECF0F1"   # status bar background
STATUS_FG   = "#2E2424"


class AnnotatorApp(tk.Tk):

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()

        self.args = args
        self.flip: bool = args.flip
        self.images_dir: Path = Path(args.images)
        self.output_path: Path = Path(args.output)

        self.data: list = []
        self.current_index: int = 0
        self.added_indices: set = set()

        # Current image order (may differ from original after swaps)
        self._current_image_order: list = []

        # Holds PhotoImage references to prevent garbage collection
        self._photo_refs: list = []

        # Widgets set during _build_ui (referenced across methods)
        self._id_label = None
        self._skill_label = None
        self._progress_label = None
        self._image_canvas = None
        self._image_inner_frame = None
        self._question_text = None
        self._choices_container = None
        self._choice_entries: list = []
        self._gt_var = None
        self._gt_row_frame = None
        self._gt_menu = None
        self._prev_btn = None
        self._next_btn = None
        self._status_label = None
        self._goto_var = None

        self.load_data()
        self._build_ui()
        self.display_item(0)

    # ------------------------------------------------------------------ #
    # Data loading                                                         #
    # ------------------------------------------------------------------ #

    def load_data(self) -> None:
        input_path = Path(self.args.input)
        if not input_path.exists():
            print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)

        all_data = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_data.append(json.loads(line))

        if not all_data:
            print(f"ERROR: Input file is empty: {input_path}", file=sys.stderr)
            sys.exit(1)

        if self.args.skills:
            allowed = {s.lower() for s in self.args.skills}
            self.data = [r for r in all_data if r.get("skill", "").lower() in allowed]
            if not self.data:
                available = sorted({r.get("skill", "") for r in all_data})
                print(
                    f"ERROR: No rows matched skills: {self.args.skills}\n"
                    f"Available skills: {available}",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            self.data = all_data

    def _build_ui(self) -> None:
        self.title("VLM Dataset Annotator")
        self.minsize(900, 750)
        self.configure(bg=BG)

        # Use clam theme
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "Nav.TButton",
            background=BTN_NAV_BG, foreground=BTN_NAV_FG,
            font=("Helvetica", 10), borderwidth=0, padding=6, relief="flat",
        )
        style.map("Nav.TButton",
                  background=[("disabled", "#A9CCE3"), ("active", "#2471A3")],
                  foreground=[("disabled", "#FDFEFE")])
        style.configure(
            "Add.TButton",
            background=BTN_ADD_BG, foreground=BTN_ADD_FG,
            font=("Helvetica", 10, "bold"), borderwidth=0, padding=6, relief="flat",
        )
        style.map("Add.TButton",
                  background=[("active", "#1E8449")])
        style.configure(
            "Swap.TButton",
            background="#DDE1E7", foreground=LABEL_FG,
            font=("Helvetica", 8), borderwidth=0, padding=3, relief="flat",
        )
        style.map("Swap.TButton",
                  background=[("active", "#BFC5CE")])

        # Main container
        main = tk.Frame(self, bg=BG)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 6))
        main.columnconfigure(0, weight=1)

        self._build_header_frame(main)
        self._build_image_frame(main)
        self._build_fields_frame(main)

        # Status bar at bottom of window
        self._status_label = tk.Label(
            self,
            text="Ready.",
            anchor=tk.W,
            relief=tk.FLAT,
            bd=0,
            padx=8,
            font=("Helvetica", 9),
            bg=STATUS_BG,
            fg=STATUS_FG,
        )
        self._status_label.pack(side=tk.BOTTOM, fill=tk.X, ipady=3)

    def _build_header_frame(self, parent: tk.Widget) -> None:
        frame = tk.Frame(parent, bg=HEADER_BG, pady=6)
        frame.grid(row=0, column=0, sticky="ew")
        frame.columnconfigure(2, weight=1)

        self._id_label = tk.Label(
            frame,
            text="",
            font=("Helvetica", 14, "bold"),
            bg=HEADER_BG,
            fg="#F39C12",
            anchor=tk.W,
            padx=10,
        )
        self._id_label.grid(row=0, column=0, sticky="w")

        self._skill_label = tk.Label(
            frame,
            text="",
            font=("Helvetica", 14, "bold"),
            bg=HEADER_BG,
            fg=HEADER_FG,
            anchor=tk.W,
            padx=10,
        )
        self._skill_label.grid(row=0, column=1, sticky="w")

        if self.flip:
            flip_lbl = tk.Label(
                frame,
                text="⟺  FLIP ON",
                font=("Helvetica", 11, "bold"),
                fg=FLIP_FG,
                bg=HEADER_BG,
            )
            flip_lbl.grid(row=0, column=2)

        self._progress_label = tk.Label(
            frame,
            text="",
            font=("Helvetica", 12),
            bg=HEADER_BG,
            fg=HEADER_FG,
            padx=10,
        )
        self._progress_label.grid(row=0, column=3, sticky="e")

    def _build_image_frame(self, parent: tk.Widget) -> None:
        outer = tk.Frame(parent, bg=IMG_BG, bd=0)
        outer.grid(row=1, column=0, sticky="nsew", pady=6)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        self._image_canvas = tk.Canvas(
            outer,
            height=IMAGE_CELL_SIZE + 20,
            bg=IMG_BG,
            highlightthickness=0,
        )
        self._image_canvas.grid(row=0, column=0, sticky="nsew")

        h_scroll = tk.Scrollbar(
            outer, orient=tk.HORIZONTAL, command=self._image_canvas.xview
        )
        h_scroll.grid(row=1, column=0, sticky="ew")
        self._image_canvas.configure(xscrollcommand=h_scroll.set)

        self._image_inner_frame = tk.Frame(self._image_canvas, bg=IMG_BG)
        self._canvas_window = self._image_canvas.create_window(
            0, 0, anchor=tk.NW, window=self._image_inner_frame
        )

        self._image_inner_frame.bind(
            "<Configure>",
            lambda e: self._image_canvas.configure(
                scrollregion=self._image_canvas.bbox("all")
            ),
        )

    def _build_fields_frame(self, parent: tk.Widget) -> None:
        frame = tk.Frame(parent, bg=BG)
        frame.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        frame.columnconfigure(0, weight=1)

        # -- Question --
        tk.Label(
            frame,
            text="Question:",
            font=("Helvetica", 11, "bold"),
            bg=BG,
            fg=LABEL_FG,
            anchor=tk.W,
        ).pack(fill=tk.X)

        q_frame = tk.Frame(frame, bg=BG)
        q_frame.pack(fill=tk.X)

        # Thin border via a 1px coloured wrapper frame
        q_border = tk.Frame(q_frame, bg="#C8CDD4", padx=1, pady=1)
        q_border.pack(side=tk.LEFT, fill=tk.X, expand=True)

        q_inner = tk.Frame(q_border, bg=ENTRY_BG)
        q_inner.pack(fill=tk.BOTH, expand=True)

        q_scroll = tk.Scrollbar(q_inner, orient=tk.VERTICAL)
        self._question_text = tk.Text(
            q_inner,
            height=5,
            wrap=tk.WORD,
            font=("Helvetica", 10),
            bg=ENTRY_BG,
            fg=ENTRY_FG,
            relief=tk.FLAT,
            bd=0,
            insertbackground=ENTRY_FG,
            yscrollcommand=q_scroll.set,
        )
        q_scroll.config(command=self._question_text.yview)
        self._question_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        q_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(
            frame,
            text="Choices:",
            font=("Helvetica", 11, "bold"),
            bg=BG,
            fg=LABEL_FG,
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(6, 0))

        self._choices_container = tk.Frame(frame, bg=BG)
        self._choices_container.pack(fill=tk.X)

        gt_label_row = tk.Frame(frame, bg=BG)
        gt_label_row.pack(fill=tk.X, pady=(6, 0))
        tk.Label(
            gt_label_row,
            text="Ground Truth:",
            font=("Helvetica", 11, "bold"),
            bg=BG,
            fg=LABEL_FG,
        ).pack(side=tk.LEFT)

        self._gt_row_frame = tk.Frame(frame, bg=BG)
        self._gt_row_frame.pack(fill=tk.X)
        self._gt_var = tk.StringVar()

        nav_frame = tk.Frame(frame, bg=BG)
        nav_frame.pack(pady=8)

        self._prev_btn = ttk.Button(
            nav_frame,
            text="<< Prev",
            command=self.go_prev,
            width=12,
            style="Nav.TButton",
        )
        self._prev_btn.pack(side=tk.LEFT, padx=8)

        self._next_btn = ttk.Button(
            nav_frame,
            text="Next >>",
            command=self.go_next,
            width=12,
            style="Nav.TButton",
        )
        self._next_btn.pack(side=tk.LEFT, padx=8)

        ttk.Button(
            nav_frame,
            text="Add",
            command=self.add_item,
            width=12,
            style="Add.TButton",
        ).pack(side=tk.LEFT, padx=8)

        tk.Frame(nav_frame, bg=BG, width=20).pack(side=tk.LEFT)  # spacer
        tk.Label(
            nav_frame,
            text="Go to:",
            font=("Helvetica", 10),
            bg=BG,
            fg=LABEL_FG,
        ).pack(side=tk.LEFT)

        goto_border = tk.Frame(nav_frame, bg="#C8CDD4", padx=1, pady=1)
        goto_border.pack(side=tk.LEFT, padx=4)
        self._goto_var = tk.StringVar()
        goto_entry = tk.Entry(
            goto_border,
            textvariable=self._goto_var,
            font=("Helvetica", 10),
            bg=ENTRY_BG,
            fg=ENTRY_FG,
            relief=tk.FLAT,
            bd=0,
            width=6,
            insertbackground=ENTRY_FG,
        )
        goto_entry.pack()
        goto_entry.bind("<Return>", lambda e: self.go_to_index())

        ttk.Button(
            nav_frame,
            text="Go",
            command=self.go_to_index,
            style="Nav.TButton",
        ).pack(side=tk.LEFT)

    def _rebuild_choice_widgets(self, choices: list) -> None:
        # Destroy existing choice entries
        for widget in self._choices_container.winfo_children():
            widget.destroy()

        self._choice_entries = []
        for i, choice_text in enumerate(choices):
            letter = CHOICE_LETTERS[i] if i < len(CHOICE_LETTERS) else str(i)
            row = tk.Frame(self._choices_container, bg=BG)
            row.pack(fill=tk.X, pady=1)

            tk.Label(
                row,
                text=f"{letter}:",
                width=3,
                anchor=tk.E,
                font=("Helvetica", 10, "bold"),
                bg=BG,
                fg=LABEL_FG,
            ).pack(side=tk.LEFT)

            e_border = tk.Frame(row, bg="#C8CDD4", padx=1, pady=1)
            e_border.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            entry = tk.Entry(
                e_border,
                font=("Helvetica", 10),
                bg=ENTRY_BG,
                fg=ENTRY_FG,
                relief=tk.FLAT,
                bd=0,
                insertbackground=ENTRY_FG,
            )
            entry.insert(0, choice_text)
            entry.pack(fill=tk.X, expand=True)
            # Bind key release to update ground truth dropdown dynamically
            entry.bind("<KeyRelease>", self._on_choice_edited)
            self._choice_entries.append(entry)

        # Rebuild OptionMenu
        if self._gt_menu is not None:
            self._gt_menu.destroy()
            self._gt_menu = None

        current_gt = self.data[self.current_index].get("ground_truth", choices[0] if choices else "")
        if current_gt not in choices and choices:
            current_gt = choices[0]
        self._gt_var.set(current_gt)

        if choices:
            self._gt_menu = tk.OptionMenu(self._gt_row_frame, self._gt_var, *choices)
            self._gt_menu.config(
                font=("Helvetica", 10),
                width=70,
                bg=ENTRY_BG,
                fg=ENTRY_FG,
                relief=tk.FLAT,
                activebackground="#D5E8F7",
                highlightthickness=0,
            )
            self._gt_menu.pack(side=tk.LEFT, padx=4)

    def _on_choice_edited(self, event) -> None:
        """Called when user edits any choice Entry - updates ground truth dropdown."""
        if not self._choice_entries:
            return

        # Get the current choices from Entry widgets
        current_choices = [entry.get() for entry in self._choice_entries]
        if not current_choices:
            return

        # Get the original choices from data to find which index was selected
        original_choices = self.data[self.current_index].get("choices", [])
        current_gt = self._gt_var.get()

        # Find the index of current ground truth in original choices
        gt_index = None
        if current_gt in original_choices:
            gt_index = original_choices.index(current_gt)

        # Rebuild the ground truth dropdown with updated choices
        if self._gt_menu is not None:
            self._gt_menu.destroy()
            self._gt_menu = None

        # If we know which choice was selected, use the updated text at that index
        if gt_index is not None and gt_index < len(current_choices):
            new_gt = current_choices[gt_index]
        elif current_gt in current_choices:
            # Current GT still exists in the new choices
            new_gt = current_gt
        else:
            # Default to first choice if current GT is not found
            new_gt = current_choices[0]

        self._gt_var.set(new_gt)

        # Rebuild the OptionMenu with current choices
        self._gt_menu = tk.OptionMenu(self._gt_row_frame, self._gt_var, *current_choices)
        self._gt_menu.config(
            font=("Helvetica", 10),
            width=70,
            bg=ENTRY_BG,
            fg=ENTRY_FG,
            relief=tk.FLAT,
            activebackground="#D5E8F7",
            highlightthickness=0,
        )
        self._gt_menu.pack(side=tk.LEFT, padx=4)

    def display_item(self, index: int) -> None:
        n = len(self.data)
        index = max(0, min(index, n - 1))
        self.current_index = index

        self.update_images(index)
        self.update_fields(index)

        row = self.data[index]
        sample_id = row.get("id", "N/A")
        skill = row.get("skill", "")
        self._id_label.config(text=f"ID: {sample_id}")
        self._skill_label.config(text=f"Skill: {skill}")
        self._progress_label.config(text=f"Item {index + 1} / {n}")

        self._prev_btn.state(["disabled"] if index == 0 else ["!disabled"])
        self._next_btn.state(["disabled"] if index == n - 1 else ["!disabled"])

    def update_images(self, index: int) -> None:
        # Reset order from source data, then render
        self._current_image_order = list(self.data[index].get("images", []))
        self._render_images()

    def _render_images(self) -> None:
        # Clear previous image widgets
        for widget in self._image_inner_frame.winfo_children():
            widget.destroy()
        self._photo_refs.clear()

        if not self._current_image_order:
            placeholder = self._make_placeholder_image("(no images)")
            lbl = tk.Label(self._image_inner_frame, image=placeholder, bg=IMG_BG)
            lbl.pack(side=tk.LEFT, padx=4, pady=4)
            self._photo_refs.append(placeholder)
            return

        n = len(self._current_image_order)
        for i, fname in enumerate(self._current_image_order):
            # Wrapper column
            col = tk.Frame(self._image_inner_frame, bg=IMG_BG)
            col.pack(side=tk.LEFT, padx=6, pady=6)

            abs_path = self.images_dir / fname
            if not abs_path.exists():
                img = self._make_placeholder_image(f"NOT FOUND:\n{fname}")
                lbl = tk.Label(
                    col,
                    image=img,
                    bg=IMG_BG,
                    highlightbackground="#E74C3C",
                    highlightthickness=2,
                )
                lbl.pack()
                self._photo_refs.append(img)
                self._set_status(f"WARNING: Image not found: {fname}")
            else:
                pil_img = Image.open(abs_path).convert("RGB")
                if self.flip:
                    pil_img = ImageOps.mirror(pil_img)
                pil_img.thumbnail((IMAGE_CELL_SIZE, IMAGE_CELL_SIZE), Image.LANCZOS)
                photo = ImageTk.PhotoImage(pil_img)
                self._photo_refs.append(photo)
                tk.Label(col, image=photo, bg=IMG_BG, bd=0, relief=tk.FLAT).pack()

            # Filename caption
            tk.Label(
                col,
                text=fname,
                font=("Helvetica", 9),
                bg=IMG_BG,
                fg="#5D6D7E",
                anchor=tk.CENTER,
            ).pack(pady=(2, 0))

            # Swap buttons (only shown when there are multiple images)
            if n > 1:
                btn_row = tk.Frame(col, bg=IMG_BG)
                btn_row.pack(pady=(3, 0))
                if i > 0:
                    ttk.Button(
                        btn_row,
                        text="← Left",
                        style="Swap.TButton",
                        command=lambda idx=i: self._swap_images(idx, idx - 1),
                    ).pack(side=tk.LEFT, padx=2)
                if i < n - 1:
                    ttk.Button(
                        btn_row,
                        text="Right →",
                        style="Swap.TButton",
                        command=lambda idx=i: self._swap_images(idx, idx + 1),
                    ).pack(side=tk.LEFT, padx=2)

        # Update canvas scroll region
        self._image_inner_frame.update_idletasks()
        self._image_canvas.configure(
            scrollregion=self._image_canvas.bbox("all")
        )

    def _swap_images(self, i: int, j: int) -> None:
        self._current_image_order[i], self._current_image_order[j] = (
            self._current_image_order[j],
            self._current_image_order[i],
        )
        self._render_images()

    def update_fields(self, index: int) -> None:
        row = self.data[index]
        choices = row.get("choices", [])

        # Update question
        self._question_text.config(state=tk.NORMAL)
        self._question_text.delete("1.0", tk.END)
        self._question_text.insert("1.0", row.get("question", ""))
        self._question_text.see(tk.END)

        # Rebuild choices and GT dropdown
        self._rebuild_choice_widgets(choices)

    def go_prev(self) -> None:
        if self.current_index > 0:
            self.display_item(self.current_index - 1)

    def go_next(self) -> None:
        if self.current_index < len(self.data) - 1:
            self.display_item(self.current_index + 1)

    def go_to_index(self) -> None:
        raw = self._goto_var.get().strip()
        if not raw:
            return
        try:
            n = int(raw)
        except ValueError:
            self._set_status(f"Invalid index: '{raw}' — enter a number between 1 and {len(self.data)}")
            return
        # Accept 1-based input (matching the "Item X / N" display)
        zero_based = n - 1
        if not (0 <= zero_based < len(self.data)):
            self._set_status(f"Index {n} out of range — valid range: 1 to {len(self.data)}")
            return
        self._goto_var.set("")
        self.display_item(zero_based)

    def add_item(self) -> None:
        row = self.data[self.current_index]

        # Collect edits
        question = self._question_text.get("1.0", tk.END).rstrip("\n")
        choices = [entry.get() for entry in self._choice_entries]
        ground_truth = self._gt_var.get()

        # Build output record (preserve all original fields, override edited ones)
        output = dict(row)
        output["question"] = question
        output["choices"] = choices
        output["ground_truth"] = ground_truth
        output["images"] = list(self._current_image_order)  # respects any reordering

        # Write to output file
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(output, ensure_ascii=False) + "\n")

        # Update tracking and status
        duplicate = self.current_index in self.added_indices
        self.added_indices.add(self.current_index)

        skill = row.get("skill", "")
        msg = f"Added item {self.current_index + 1} (skill: {skill}) to {self.output_path}"
        if duplicate:
            msg = "WARNING: duplicate — " + msg
        self._set_status(msg)

    def _set_status(self, message: str) -> None:
        self._status_label.config(text=message)

    def _make_placeholder_image(self, label_text: str) -> ImageTk.PhotoImage:
        size = IMAGE_CELL_SIZE
        img = Image.new("RGB", (size, size), color=(180, 180, 180))
        draw = ImageDraw.Draw(img)
        # Draw centered text (simple approach without loading fonts)
        lines = label_text.split("\n")
        line_h = 16
        total_h = line_h * len(lines)
        y_start = (size - total_h) // 2
        for i, line in enumerate(lines):
            # Estimate text width: ~7px per char
            text_w = len(line) * 7
            x = max(0, (size - text_w) // 2)
            draw.text((x, y_start + i * line_h), line, fill=(80, 80, 80))
        return ImageTk.PhotoImage(img)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tkinter GUI annotator for VLM benchmark JSONL datasets."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSONL file (e.g. data/bench_full_fidel.jsonl)",
    )
    parser.add_argument(
        "--images",
        required=True,
        help="Path to folder containing images (e.g. data/images/)",
    )
    parser.add_argument(
        "--output",
        default="annotated.jsonl",
        help="Path to output JSONL file (default: annotated.jsonl)",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help=(
            "Horizontally flip images in the display (visual only). "
            "Output JSONL always saves original filenames unchanged."
        ),
    )
    parser.add_argument(
        "--skills",
        nargs="+",
        default=None,
        metavar="SKILL",
        help=(
            "Only display rows whose 'skill' matches one of these values "
            "(space-separated, case-insensitive). "
            "Example: --skills navigation relative_agents"
        ),
    )
    args = parser.parse_args()

    app = AnnotatorApp(args)
    app.mainloop()


if __name__ == "__main__":
    main()
