# -*- coding: utf-8 -*-
#2022.4.20
#二进制文件转化为图像

import os, math
from PIL import Image




        

def getBinaryData(filename):
        '''
        Extract byte values from binary executable file and store them into list
        param filename: PE executable file name
        return: byte value list
        '''
        binary_values = []

        with open(filename, 'rb') as fileobject:

            # read file byte by byte
            data = fileobject.read(1)

            while data != b'':
                binary_values.append(ord(data))
                data = fileobject.read(1)
        
        fileobject.close

        return binary_values

def createGreyScaleImage(greyscale_data,width=None):
        """
        Create greyscale image from binary data. Use given with if defined or create square size image from binary data.
        :param filename: image filename
        """
        #greyscale_data  = self.getBinaryData()
        size            = get_size(len(greyscale_data), width)
        img=create_file(greyscale_data, size, 'L')
        
        return img 


def get_size(data_length, width=None):
        """
        Returns the dimesnions of the image according to the data size of the file, inspired from visualization and automatic classification by L. Nataraj (http://dl.acm.org/citation.cfm?id=2016908)
        :param data_length: the number of bytes of the executable file
        :param width: the desired width for the image
        :return: (width of the image, height of the image)
        """
        if width is None: # with don't specified any with value

            size = data_length

            if (size < 10240):
                width = 32
            elif (10240 <= size <= 10240 * 3):
                width = 64
            elif (10240 * 3 <= size <= 10240 * 6):
                width = 128
            elif (10240 * 6 <= size <= 10240 * 10):
                width = 256
            elif (10240 * 10 <= size <= 10240 * 20):
                width = 384
            elif (10240 * 20 <= size <= 10240 * 50):
                width = 512
            elif (10240 * 50 <= size <= 1024 * 1024):
                width = 768
            elif(1024 * 1024 <= size <= 1024 * 1024*10):
                width = 1024
            elif(1024 * 1024*10 <= size <= 1024 * 1024*20):
                width = 1536
            elif(1024 * 1024*20 <= size <= 1024 * 1024*30):
                width = 2048
            elif(1024 * 1024*30 <= size <= 1024 * 1024*40):
                width = 2560
            elif(1024 * 1024*40 <= size <= 1024 * 1024*70):
                width = 3252
            else:
                width=4096
                
            height = int(size / width) + 1

        else:
            width  = int(math.sqrt(data_length)) + 1
            height = width

        return (width, height)


def create_file(data, size, image_type):
        """
        create the file with filename and image_type as an image 
        :param data: the data of the image
        :param size: the total size of the image
        :param image_type: the type of the image added to the filename for the image name
        :return: None
        """
        try:
            image = Image.new(image_type, size)
            image.putdata(data)
            return image

        except Exception as err:
            print(err)

    