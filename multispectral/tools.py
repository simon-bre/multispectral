import os
import cv2
import re

class Tools:
    """Just some useful things."""

    #recursively transcodes a file tree
    @staticmethod
    def transcode(format, dir_in, dir_out, regex='.', enc_param=None):
        for d, _, files in os.walk(dir_in):
            for f in [f for f in files if re.search(regex, f, re.IGNORECASE)]:
                f_in = os.path.join(d, f)
                im = cv2.imread(f_in)
                if im is None:
                    continue

                basename = os.path.splitext(f)[0]
                format = format.replace('.', '')
                d_out = d.replace(dir_in, dir_out)
                if not os.path.exists(d_out):
                    os.makedirs(d_out)
                f_out = os.path.join(d_out, basename+'.'+format)

                if format.lower() in ['jpg', 'jpeg']:
                    if enc_param is None:
                        enc_param = 95
                    cv2.imwrite(f_out, im, (cv2.IMWRITE_JPEG_QUALITY, enc_param))
                elif format.lower() == 'png':
                    if enc_param is None:
                        enc_param = 3
                    cv2.imwrite(f_out, im, (cv2.IMWRITE_PNG_COMPRESSION, enc_param))

                print('Transcoded: %s' % f_out)

    @staticmethod
    def add_suffix(file, suffix):
        file_base, file_ext = os.path.splitext(file)
        return file_base + suffix + file_ext
