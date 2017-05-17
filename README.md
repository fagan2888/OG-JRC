# JRC Overlapping Generations Model Training by Open Research Group

This public repository contains the training materials, tutorials, and code for the two week training delivered to the Joint Research Centre of the European Commission in Seville, Spain, May 29 to June 9, 2017 by [Richard Evans](https://sites.google.com/site/rickecon/) and [Jason DeBacker](http://jasondebacker.com/) of [Open Research Group, Inc.](http://openrg.com/) (OpenRG).

We are planning on the following schedule each of the 10 workdays during that time. Let us know if we need to change this schedule in any way.

* 9am to noon: Lecture, theory, computational instruction
* noon to 1pm: Lunch
* 1pm to 4pm: Guided computational practice, implementation, problem sets

We have created a series of textbook chapters complete with theoretical exposition, computational description and tips, and problem sets. These chapters and their instruction are meant to progressively build upon each other. In the end, you will be building a computational implementation of an OG model for fiscal policy that is thousands of lines of code. We will train your research group to understand each code section and to write the code in a way that is accessible, modular, scalable, and amenable to collaboration among a large group of researchers.

This will be an intensive two weeks. We are providing your researchers 7 areas of tutorials that they will benefit from reading before the training. We will, of course, teach these things as we go through the material. But we will be able to proceed at a faster pace if the attendees are already familiar with most of the concepts below.

Pre-course Tutorial Areas

1. Instructions for installing the Anaconda distribution of Python
2. Text editor suggestions (Atom, Sublime Text 3, Vim)
3. PEP8, docstring commenting, and module structure
4. Git and GitHub tutorial
5. Jupyter Notebooks
6. Basic Python tutorials (data structures, logic, functions and modules)
7. Intermediate Python tutorials (pandas, root finders and minimizers)


## 1. Instructions for installing the Anaconda distribution of Python

We will be using the [Python](https://www.python.org/) programming language and many of its powerful libraries for writing the code that will run the overlapping generations models in this training. Using an open source language, such as Python, has the advantage of being free and accessible for anyone who wishes to contribute to this project. Being open source also allows Python users to go into the source code of any function to modify it to suit one's needs.

We recommend that each participant download the Anaconda distribution of Python provided by [Continuum Analytics](https://www.continuum.io/). We recommend the most recent stable version of Python, which is currently Python 3.6. This can be done from the [Anaconda download page](https://www.continuum.io/downloads) for Windows, Mac OSX, and Linux machines. The code we will be writing uses common Python libraries such as `NumPy`, `SciPy`, `pickle`, `os`, `matplotlib`, and `time`.


## 2. Text editor suggestions

You will want a capable text editor for developing your code.  There are three we recommend: [Atom](https://atom.io), [Sublime Text](https://www.sublimetext.com), and [Vim](http://www.vim.org).  Atom and Vim are completely Free, a trial version of Sublime Text is available for free, but a licensed version is $70.
[TODO: Add instruction about Atom, Sublime Text 3, and Vim.]

### 2.1. Atom

There are several packages you'll want to install with Atom.  Once Atom is installed, you can add packages by navigating Atom->Preferences->Install and then typing in the name of the package you would like to install.  

For work with Python, we recommend the following packages be installed:

* MagicPython
* python-indent
* tabs-to-spaces
* minimap
* open-recent

For development with GitHub we recommend:

* merge-conflict

If using LaTex in this editor, the following packages are helpful:

* atom-latex
* latextools
* autocomplete-bitex
* dictionary
* latexer
* pdf-view

In addition, you will also want to download the [Skim](http://skim-app.sourceforge.net) pdf viewer to aid in displaying pdf's compiled from TeX with Atom.

### 2.2 Sublime Text

### 2.3 Vim

Vim is free and very powerful.  It does, however, have a very steep learning curve - using keyboard shortcuts for all commands.  


## 3. PEP8, docstring commenting, and module structure

[TODO: Add instruction here. Reference the [PythonFuncs.ipynb](https://github.com/OpenRG/OG-JRC/blob/master/Tutorials/PythonFuncs.ipynb) Jupyter notebook.]


## 4. Git and GitHub tutorial

We have included a tutorial on using [Git and GitHub.com](https://github.com/OpenRG/OG-JRC/blob/master/Tutorials/git_tutorial.pdf) in the [Tutorials](https://github.com/OpenRG/OG-JRC/tree/master/Tutorials) directory of this repository.


## 5. Jupyter Notebooks

[Jupyter notebooks](http://jupyter.org/) are files that end with the `*.ipynb` suffix. These notebooks are opened in a browser environment and are an open source web application that combines instructional text with live executable and modifyable code for many different programming platforms (e.g., Python, R, Julia). Jupyter notebooks are an ideal tool for teaching programming as they provide the code for a user to execute and they also provide the context and explanation for the code. We have provided a number of Jupyter notebooks in the [Tutorials](https://github.com/OpenRG/OG-JRC/tree/master/Tutorials) folder of this repository.

These notebooks used to be Python-specific, and were therefore called iPython notebooks (hence the `*.ipynb` suffix). But Jupyter notebooks now support many programming languages, although the name still pays homage to Python with the vestigal "py" in "Jupyter". The notebooks execute code from the kernel of the specific programming language on your local machine.

Jupyter notebooks capability will be automatically installed with your download of the [Anaconda distribution](https://www.continuum.io/downloads) of Python. If you did not download the Anaconda distribution of Python, you can download Jupyter notebooks separately by following the instructions on the Jupyter [install page](http://jupyter.org/install.html).


### 5.1. Opening a Jupyter notebook

Once Jupyter is installed--whether through Anaconda or through the Jupyter website--you can open a Jupyter notebook by the following steps.

1. Navigate in your terminal to the folder in which the Jupyter notebook files reside. In the case of the Jupyter notebook tutorials in this repository, you would navigate to the `~/OG-JRC/Tutorials/` directory.
2. Type `jupyter notebook` at the terminal prompt.
3. A Jupyter notebook session will open in your browser, showing the available `*.ipynb` files in that directory.
4. Double click on the Jupyter notebook you would like to open.

It is worth noting that you can also simply navigate to the URL of the Jupyter notebook file in the GitHub repository on the web (e.g., [https://github.com/OpenRG/OG-JRC/blob/master/Tutorials/PythonReadIn.ipynb](https://github.com/OpenRG/OG-JRC/blob/master/Tutorials/PythonReadIn.ipynb)). You can read the Jupyter notebook on GitHub.com, but you cannot execute any of the cells. You can only execute the cells in the Jupyter notebook when you follow the steps above and open the file from your terminal.


### 5.2. Using an open Jupyter notebook

Once you have opened a Jupyter notebook, you will find the notebook has two main types of cells: Markdown cells and Code cells. Markdown cells have formatted Jupyter notebook markdown text, and serve primarily to present context for the coding cells. A reference for the markdown options in Jupyter notebooks is found in the [Jupyter markdown documentation page](http://jupyter-notebook.readthedocs.io/en/latest/examples/Notebook/Working%20With%20Markdown%20Cells.html).

You can edit a Markdown cell in a Jupyter notebook by double clicking on the cell and then making your changes. Make sure the cell-type box in the middle of the top menu bar is set to `Markdown`. To implement your changes in the Markdown cell, type `Shift-Enter`.

A Code cell will have a `In [ ]:` immediately to the left of the cell for input. The code in that cell can be executed by typing `Shift-Enter`. For a Code cell, the  cell-type box in the middle of the top menu bar says `Code`.


### 5.3. Closing a Jupyter notebook

When you are done with a Jupyter notebook, you first save any changes that you want to remain with the notebook. Then you close the browser windows associated with that Jupyter notebook session. You must then close the local server that was opened to run the Jupyter notebook in your terminal window. On a Mac or Windows, this is done by going to your terminal window and typing `Ctrl-C` and then selecting `y` for yes and hitting `Enter`.


## 6. Basic Python tutorials

For this training, we have included in this repository five basic Python tutorials in the [Tutorials](https://github.com/OpenRG/OG-JRC/tree/master/Tutorials) directory.

1. [PythonReadIn.ipynb](https://github.com/OpenRG/OG-JRC/blob/master/Tutorials/PythonReadIn.ipynb). This Jupyter notebook ...
2. [PythonNumpyPandas.ipynb](https://github.com/OpenRG/OG-JRC/blob/master/Tutorials/PythonNumpyPandas.ipynb). This Jupyter notebook ...
3. [PythonDescribe.ipynb](https://github.com/OpenRG/OG-JRC/blob/master/Tutorials/PythonDescribe.ipynb). This Jupyter notebook ...
4. [PythonFuncs.ipynb](https://github.com/OpenRG/OG-JRC/blob/master/Tutorials/PythonFuncs.ipynb). This Jupyter notebook ...
5. [PythonVisualize.ipynb](https://github.com/OpenRG/OG-JRC/blob/master/Tutorials/PythonVisualize.ipynb). This Jupyter notebook ...

## 7. Intermediate Python tutorials

[TODO: Add root finder and minimization Jupyter notebook.]
