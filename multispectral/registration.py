import os
import re
import shutil
import SimpleITK as sitk
import warnings
from multispectral import Frame, Layer, Tools
import numpy as np
import cv2

class Registration:
    """ Contains functions for parametric feature based and non-parametric (deformable) registration"""

    @staticmethod
    def register_coarse(frame_ref, frame_moving,
                        regex_master_ref='', regex_master_moving='',
                        dir_out='registered_coarse', suffix_out='_creg',
                        verbose=False):
        """
        Registers one frame to another using a good old perspective projection from RANSAC'd matching SIFT features.
        The transformation is computed for one layer from the two frames respectively and applied to all other layers.
        :param frame_ref: frame that stays fixed
        :param frame_moving: frame to be transformed
        :param regex_master_ref: layer of frame_ref used for computing the transformation
        :param regex_master_moving: layer of frame_moving used for computing the transformation
        :param dir_out: output directory for registered images. Can be absolute or relative (to frame_ref.root_dir)
        :param suffix_out: suffix that will be attached to original filename to indicate its registration status
        :param verbose: if True, registration success is visualized. only for testing (waits for input after each image)
        :return: Frame containing now registered frame_moving
        """
        #TODO: implement
        warnings.warn('register_coarse IS NOT IMPLEMENTED YET and returns None.')
        return None

    @staticmethod
    def register_fine(frame, regex='', regex_ref='', dir_out='registered_fine', suffix_out='_freg', verbose=False):
        """
        Inter-registers all relevant images of frame, using a non-parametric deformable transformation.
        Reference is first image matching regex_ref. This function needs elastix binaries installed on your machine.
        http://elastix.isi.uu.nl/
        :param frame: input frame
        :param regex: filter for input images
        :param regex_ref: regex for reference image (first one encountered is taken)
        :param dir_out: output directory for registered images. Can be absolute or relative (to frame.root_dir)
        :param suffix_out: suffix that will be attached to original filename to indicate its registration status
        :param verbose: if True, registration success is visualized. only for testing (waits for input after each image) (TODO)
        :return: Frame of now registered images
        """

        #find and load reference layer
        ref_layer: Layer = [la for la in frame.layers if re.search(regex_ref, os.path.split(la.file)[1])][0]
        ref_img = cv2.imread(ref_layer.file, cv2.IMREAD_GRAYSCALE)
        ref_img_sitk = sitk.GetImageFromArray(ref_img)
        #ref_img_sitk = sitk.ReadImage(ref_layer.file)
        #ref_img_sitk.SetSpacing((1.0, 1.0))     #safety
        print('loaded %s as reference layer' % ref_layer.file)

        #make output directory and put reference image there
        if not os.path.isabs(dir_out):
            dir_out = os.path.join(frame.root_dir, dir_out)
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        file_ref_out = os.path.join(
            dir_out,
            Tools.add_suffix(os.path.split(ref_layer.file)[1], suffix_out)
        )
        shutil.copy(ref_layer.file, file_ref_out)    #TODO: make sure this is not in another format that other registered images..

        #make output frame and put reference image there
        frame_out = Frame(name=frame.name+suffix_out, root_dir=frame.root_dir) #root dir is the same as for unregistered frame. yes, this makes sense.
        frame_out.append(Layer(name=ref_layer.name, file=file_ref_out))


        for layer in [la for la in frame.layers if re.search(regex, os.path.split(la.file)[1]) and la is not ref_layer]:
            print('registering %s...' % layer.file)
            moving_img = cv2.imread(layer.file, cv2.IMREAD_GRAYSCALE)
            moving_img_sitk = sitk.GetImageFromArray(moving_img)
            #moving_img_sitk = sitk.ReadImage(layer.file)
            #moving_img_sitk.SetSpacing((1.0, 1.0))

            #create registration object and add image pair
            filter = sitk.ElastixImageFilter()
            filter.SetFixedImage(ref_img_sitk)
            filter.SetMovingImage(moving_img_sitk)

            #define parameters
            params = sitk.GetDefaultParameterMap("bspline")
            params['MaximumNumberOfIterations'] = ['512']
            params['FinalGridSpacingInPhysicalUnits'] = ['100']
            #params_b['FinalGridSpacingInPhysicalUnits'] = ['300']
            #TODO: make this non-hardcoded, or dependent of resolution..

            filter.SetParameterMap(params)
            filter.PrintParameterMap()
            #TODO: as we apply the registration to only one image here, we could also use the default multi-resolution approach here..
            # ("ElastixImageFilter will register our images with a translation -> affine -> b-spline multi-resolution approach by default.")

            # do registration
            filter.Execute()

            #save registered image
            result_img_sitk = filter.GetResultImage()
            file_out = os.path.join(
                dir_out,
                Tools.add_suffix(os.path.split(layer.file)[1], suffix_out)
            )
            #sitk.WriteImage(result_img_sitk, file_out)  #somehow this doesn't produce proper images..
            result_img = sitk.GetArrayFromImage(result_img_sitk)
            if np.max(result_img) >= 2**8:
                result_img = np.uint16(result_img)
            else:
                result_img = np.uint8(result_img)
            cv2.imwrite(file_out, result_img)

            #append to result frame
            frame_out.append(Layer(name=layer.name, file=file_out))

            #resultimg = np.uint16(sitk.GetArrayFromImage(resultimg_sitk))
            # f_out = os.path.join(patchdir_out, masterfiles[0])
            # f_out = f_out.replace(code_in, code_out)
            # cv2.imwrite(f_out, resultimg)

            # make composite image for easy quality control
            # black = refimg * 0
            # composite = cv2.merge((refimg, black, resultimg))
            # comp_dir = os.path.join(dir_out, 'composite')
            # if not os.path.exists(comp_dir):
            #     os.makedirs(comp_dir)
            # cv2.imwrite(os.path.join(comp_dir, 'comp_'+masterfiles[0]), composite)
            # if verbose:
            #     show(composite)

        return frame_out
