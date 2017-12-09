import tkinter
import subprocess
import sys
from tkinter import filedialog


class GUI(object):
    def __init__(self, parent):
        self.parent = parent
        self.frame = tkinter.Frame(parent)
        self.frame.pack()

        self.btn1 = tkinter.Button(
            self.frame, text='Choose train directory', command=train)
        self.btn1.pack()

        self.btn2 = tkinter.Button(
            self.frame, text='Choose test file', command=test_single_data)
        self.btn2.pack()

        self.btn3 = tkinter.Button(
            self.frame, text='Choose test directory', command=test_dataset)
        self.btn3.pack()

        self.scroll = tkinter.Scrollbar(self.frame)
        self.scroll.pack(side=tkinter.RIGHT, fill=tkinter.Y)
        self.text = tkinter.Text(self.frame, height=40, width=80)
        self.text.pack(side=tkinter.LEFT, fill=tkinter.Y)

        self.scroll.config(command=self.text.yview)
        self.text.config(yscrollcommand=self.scroll.set)


def train():
    train_dir = filedialog.askdirectory()
    p = subprocess.Popen(
        ['python', 'train.py', train_dir],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = p.communicate()
    gui.text.insert(tkinter.END, output)


def test_single_data():
    test_file = filedialog.askopenfile()
    p = subprocess.Popen(
        ['python', 'test.py', test_file.name],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = p.communicate()
    gui.text.insert(tkinter.END, output)


def test_dataset():
    test_dir = filedialog.askdirectory()
    p = subprocess.Popen(
        ['python', 'test.py', test_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    output, errors = p.communicate()
    gui.text.insert(tkinter.END, output)

root = tkinter.Tk()
root.title('Simple text classification')
gui = GUI(root)
root.mainloop()
