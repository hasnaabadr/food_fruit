# gui_tests.py
import tkinter as tk
from tkinter import filedialog, messagebox
from all_tests import (test_partB_case1, test_partB_case2 ,  test_partC
#    test_partA,
#    test_partD, test_partE, integrated_test
)
from part5test import test_random_images  


# ==========================
# Main GUI
# ==========================
root = tk.Tk()
root.title("Food & Fruit Test GUI")
root.geometry("500x400")

# ======= Functions for Buttons =======

def run_test_a():
    test_file = filedialog.askopenfilename(title="Select image for Part A")
    # if test_file:
        # result = test_partA(test_file)
        # messagebox.showinfo("Result Part A", str(result))

def run_test_b1():
    test_dir = filedialog.askdirectory(title="Select test images folder for Part B - Case 1")
    if test_dir:
        results = test_partB_case1(test_dir)
        # Save results to a text file
        with open("partB_case1_results.txt", "w") as f:
            for img_name, pred_class in results:
                f.write(f"{img_name}: {pred_class}\n")        
                messagebox.showinfo(
                    "Result Part B Case 1",
            f"Results saved to partB_case1_results.txt\nTotal images: {len(results)}"
        )


def run_test_b2():
    anchor_dir = filedialog.askdirectory(title="Select folder containing Anchor and other images for Part B - Case 2")
    if anchor_dir:
        results = test_partB_case2(anchor_dir)  
        messagebox.showinfo(
            "Result Part B Case 2",
            f"Results saved to partB_case2_results.txt\nMost similar image: {min(results, key=results.get)}"
        )

def run_test_c():
    test_dir = filedialog.askdirectory(title="Select test images folder for Part C")
    if test_dir:
       
        results = test_partC(test_dir)
        messagebox.showinfo(
            "Result Part C",
            f"Results saved to partC_results.txt\nTotal images: {len(results)}"
        )


def run_test_d():
    test_dir = filedialog.askdirectory(title="Select test images folder for Part D")
    # if test_dir:
    #     result = test_partD(test_dir)
    #     messagebox.showinfo("Result Part D", str(result))

def run_test_e():
    test_dir = filedialog.askdirectory(title="Select test images folder for Part E")
    # if test_dir:
    #     result = test_partE(test_dir)
    #     messagebox.showinfo("Result Part E", str(result))

def run_integrated():
    test_dir = filedialog.askdirectory(title="Select test images folder for Integrated Test")
    # if test_dir:
    #     result = integrated_test(test_dir)
    #     messagebox.showinfo("Integrated Test Result", str(result))

# ======= Buttons =======
buttons = [
    ("Test Part A", run_test_a),
    ("Test Part B - Case 1", run_test_b1),
    ("Test Part B - Case 2", run_test_b2),
    ("Test Part C", run_test_c),
    ("Test Part D", run_test_d),
    ("Test Part E", run_test_e),
    ("Integrated Test", run_integrated)
]

for i, (text, func) in enumerate(buttons):
    btn = tk.Button(root, text=text, width=30, height=2, command=func)
    btn.pack(pady=5)

# ==========================
root.mainloop()
