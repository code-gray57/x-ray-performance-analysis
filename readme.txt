
--------------------------------------------------
-----     IEEE/ANSI N42.55 Analysis code   -------
-----        Developed by Jack Glover      -------
-----        email  firstname.lastname     -------
-----               at    nist  gov        -------
--------------------------------------------------


This code was written to process images of the IEEE/ANSI N42.55 test object. It processes the images according to the methods in the standard, computes scores for all metrics, and produces a report written in LaTeX. If you have LaTeX installed, it will compile the tex file into a pdf.

WARNING: This code is still under active development and is a work in progress. It is not yet ready for public release and is not user friendly at all. Please email Jack with any bugs or suggestions.

Nominal version: 0.11
Code last modified: Oct 4th, 2017


-----    Python version   -------

I have tested this code on Python 2.7 and 3.5 and it works on both of them. It likely works on other versions too.

-----    Required libraries   -------

This code requires numerous libraries. Some of the most important are: numpy, scipy, opencv-python, scikit-image, matplotlib.

NOTE: I should update this list to be complete in the future and add library version numbers.

If you get error messages installing any of these libraries, and are using a windows machine, then consider installing from the precompiled binaries available at http://www.lfd.uci.edu/~gohlke/pythonlibs/

If you want the final report to be complied to a pdf, then you must have a LaTeX installation (possibly pdflatex).

-----    Format of images   -------

The required format for the images can be seen in the example folder.
