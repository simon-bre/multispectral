# multispectral

This is a convenience package for basic operations on multispectral images, where the spectral layers are stored as individual image files within some folder. The package was designed with the applilcation area of cultural heritage / manuscript studies in mind.

**CAUTION: very prototypy at the moment, would not recommend use.. yet :)**

##current features:
* easy data handling with "Frames" and regular expressions
* deformable fine registration using [elastix](http://elastix.isi.uu.nl/)
* decomposition/clustering using [scikit-learn](https://scikit-learn.org/)

##requirements:
* opencv-python
* numpy
* scipy
* scikit-learn
* [SimpleElastix](https://pypi.org/project/SimpleElastix/) **WARNING: THIS WILL NOT BE ENFORCED BY pip**

##usage:

Suppose you have your multispectral layers in a folder `'/somepath/codexX-pageY'` 
(or any of its subfolders), and the files look something like:
`'codexX-pageY_400nm.tif', 'codexX-pageY_500nm.tif',...`

Then you could go:
```python
from multispectral import Frame,Layer,Registration,Unmixing

# collect images in root_dir matching regex; groups 1 and 2 of the match object 
# identify the document and the layer respectively (optional)
frame = Frame(root_dir='/somepath/codexX-pageY',
                regex='(.+-.+)_(\d{3}nm).tif',
                group_framename=1,
                group_layername=2)
                
# inter-register all layers (regex_ref defines the fixed 'reference image',
# store the result in a given output folder (or by default frame.root_dir/registered_fine),
# and return a frame containing those resulting images
registered = Registration.register_fine(frame=frame, regex_ref='500nm')

# make unmixing object: loads images of frame and converts them to a data matrix
um = Unmixing(registered)
# perform principal component analysis, store visualizations of first 5 components
# in given output folder (or by default frame.root_dir/pca), return frame containing those
principal_components = um.unmix(method=Unmixing.Method.PCA, n_components=5, p_keep=0.5)
```

Simple as that.
