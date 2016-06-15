import sys
import os
import math
import numpy as np
from scipy.ndimage.filters import gaussian_filter

def resolution_generator(x, y, res_sizes):
    x = x - res_sizes[0] / 2
    y = y - res_sizes[1] / 2
    return x * x + y * y

def ctf_xy_generator(x, y, nx, angpix):
    xs = nx * angpix
    rx = (x - nx / 2) / xs
    ry = (y - nx / 2) / xs
    return (rx, ry)

class mrc(object):
    def __init__(self, x = 0, y = 0, z = 0, data = 0):
        self.nx = x
        self.ny = y
        self.nz = z
        self.data = data
        self.data_min = 0
        self.data_max = 0
        self.data_avg = 0
        self.data_stddev = 0
        self.angpix = 0
        self.defocus_u = 0
        self.defocus_v = 0
        self.defocus_angle = 0
        self.voltage = 0
        self.cs = 0
        self.q0 = 0
        self.bfac = 0
        self.defocus_average = 0
        self.defocus_deviation = 0
        self.lmbda = 0
        self.K1 = 0
        self.K2 = 0
        self.K3 = 0
        self.K4 = 0
        if self.data:
            self.updateStatistics()

    def copyFromMRC(self, m):
        self.nx = m.nx
        self.ny = m.ny
        self.nz = m.nz
        self.data = m.data.copy()
        self.data_min = m.data_min
        self.data_max = m.data_max
        self.data_avg = m.data_avg
        self.data_stddev = m.data_stddev
        self.angpix = m.angpix
        self.defocus_u = m.defocus_u
        self.defocus_v = m.defocus_v
        self.defocus_angle = m.defocus_angle
        self.voltage = m.voltage
        self.cs = m.cs
        self.q0 = m.q0
        self.bfac = m.bfac
        self.defocus_average = m.defocus_average
        self.defocus_deviation = m.defocus_deviation
        self.lmbda = m.lmbda
        self.K1 = m.K1
        self.K2 = m.K2
        self.K3 = m.K3
        self.K4 = m.K4
        self.updateStatistics()

    def readFromFile(self, filename, startSlice=1, numSlices=1000000000):
        with open(filename, 'r') as file:
            a = np.fromfile(file, dtype=np.int32, count=10)
            shouldSwap = 0
            if abs(a[0] > 100000):
                a.byteswap()
                shouldSwap = 1
            
            mode = a[3]
            
            b = np.fromfile(file, dtype=np.float32, count=12)
            if shouldSwap:
                b.byteswap()
    
            mi = b[9]
            ma = b[10]
            mv = b[11]
            
            
            c = np.fromfile(file, dtype=np.int32, count=30)
            if shouldSwap:
                c.byteswap()
            
            d = np.fromfile(file, dtype=np.uint8, count=8)
            if shouldSwap:
                d.byteswap()
            
            e = np.fromfile(file, dtype=np.int32, count = 2)
            if shouldSwap:
                e.byteswap()
            
            ns = min(e[1],10)
            for i in range(1,11):
                g = np.fromfile(file, dtype=np.uint8, count = 80)
        
            self.nx = a[0]
            self.ny = a[1]
            self.nz = a[2]
            
            datatype = np.float32
            if mode == 0:
                datatype = np.int8
            elif mode == 1:
                datatype = np.int16
            elif mode == 2:
                datatype = np.float32
            elif mode == 6:
                datatype = np.uint16
            
            
            if c[1] > 0:
                extraHeader = np.fromfile(file, dtype=np.uint8, count = c[1])
            
            nz = self.nz
            if startSlice > 1:
                discard = np.fromfile(file, dtype=datatype, count = (startSlice - 1) * self.nx * self.ny)
                nz = min(self.nz - (startSlice - 1), numSlices)

            self.nz = nz
            ndata = self.nx * self.ny * self.nz

            originalData = np.fromfile(file, dtype=datatype, count=ndata)
            if shouldSwap:
                originalData.byteswap()
            if self.nz > 1:
                originalData.resize((self.nx, self.ny, self.nz))
            elif self.nz == 1:
                originalData.resize((self.nx, self.ny))
            self.data = originalData.astype(np.float32)
            self.updateStatistics()

    def updateStatistics(self):
        self.data_mean = self.data.mean()
        self.data_stddev = self.data.std()
        self.data_min = self.data.min()
        self.data_max = self.data.max()
        return (self.data_mean, self.data_stddev, self.data_min, self.data_max)
    
    def calculateStatistics(self):
        return (self.data_mean, self.data_stddev, self.data_min, self.data_max)
    
    def getImageContrast(self, sigmaContrast):
        (avg, stddev, minval, maxval) = self.calculateStatistics()
        if sigmaContrast > 0:
            minval = avg - sigmaContrast * stddev
            maxval = avg + sigmaContrast * stddev
            self.data = np.clip(self.data, minval, maxval)
            self.updateStatistics()

    def getImageContrastAdjustMean(self, sigmaContrast, meanSigmas):
        (avg, stddev, minval, maxval) = self.calculateStatistics()
        avg += meanSigmas * stddev
        if sigmaContrast > 0:
            minval = avg - sigmaContrast * stddev
            maxval = avg + sigmaContrast * stddev
            self.data = np.clip(self.data, minval, maxval)
            self.updateStatistics()


    def extract2DBox(self, x, y, z, boxsize):
        #swap x and y
        oldx = x
        x = y
        y = oldx
        (avg, stddev, minval, maxval) = self.calculateStatistics()
        range = maxval - minval
        step = range / 255.0
        
        xo = x - boxsize / 2
        yo = y - boxsize / 2
        
        if ((xo >= 0) and (xo + boxsize < self.nx) and (yo >= 0) and (yo + boxsize < self.ny)):
            newData = self.data[xo:xo+boxsize, yo:yo+boxsize]
        else:
            extractX = boxsize
            extractY = boxsize
            offsetX = 0
            offsetY = 0
            if xo < 0:
                extractX += xo
                offsetX = xo * -1
                xo = 0
            elif xo + boxsize > self.nx:
                extractX = self.nx - xo
            if yo < 0:
                extractY += yo
                offsetY = yo * -1
                yo = 0
            elif yo + boxsize > self.ny:
                extractY = self.ny - yo
            #            print('Xo: ' + str(xo) + ' exX: ' + str(extractX) + ' ofX: ' + str(offsetX) + ' Yo: ' + str(yo) + ' exY: ' + str(extractY) + ' ofY: ' + str(offsetY))
            box = self.data[xo:xo+extractX, yo:yo+extractY]
            newData = np.ndarray(shape=(boxsize,boxsize), dtype=np.float32)
            newData.fill(minval + range/2);
            newData[offsetX:offsetX+extractX, offsetY:offsetY+extractY] = box
        
        return mrc(boxsize, boxsize, 1, newData)

    def x(self):
        return self.nx
    
    def y(self):
        return self.ny
    
    def get2DPoint(self, x, y):
        return self.data[x, y, 0]
    
    def getScaled2DData(self):
        (avg, stddev, minval, maxval) = self.calculateStatistics()
        dataCopy = self.data.reshape((self.nx, self.ny)).copy()
        range = maxval - minval
        dataCopy = (dataCopy - minval) / range - 0.5
        return dataCopy

    #   Better to do this and use the contrast of the entire image, rather than just extracted box...
#        Clamp to range [-0.5, 0.5]
    def generateScaled2DBox(self, xc, yc, boxsize):
        #swap x and y
        oldx = xc
        xc = yc
        yc = oldx
        (avg, stddev, minval, maxval) = self.calculateStatistics()
        range = maxval - minval
        step = range / 255.0
        
        xo = xc - boxsize / 2
        yo = yc - boxsize / 2
        
        if ((xo >= 0) and (xo + boxsize < self.nx) and (yo >= 0) and (yo + boxsize < self.ny)):
            newData = self.data[xo:xo+boxsize, yo:yo+boxsize]
        else:
            extractX = boxsize
            extractY = boxsize
            offsetX = 0
            offsetY = 0
            if xo < 0:
                extractX += xo
                offsetX = xo * -1
                xo = 0
            elif xo + boxsize > self.nx:
                extractX = self.nx - xo
            if yo < 0:
                extractY += yo
                offsetY = yo * -1
                yo = 0
            elif yo + boxsize > self.ny:
                extractY = self.ny - yo
            box = self.data[xo:xo+extractX, yo:yo+extractY]
            newData = np.ndarray(shape=(boxsize,boxsize), dtype=np.float32)
            newData.fill(minval + range/2);
            newData[offsetX:offsetX+extractX, offsetY:offsetY+extractY] = box


        newDataCopy = newData.reshape((boxsize, boxsize)).copy()
        newDataCopy = (newDataCopy - minval) / range - 0.5
        return newDataCopy


    def lowpass_filter(self, angpix, low_pass, filter_edge_width = 2):
        ft = np.fft.fftshift(np.fft.fft2(self.data))
        ori_size = self.nx
        ires_filter = int(round((ori_size * angpix) / low_pass))
        filter_edge_halfwidth = filter_edge_width / 2

        edge_low = (ires_filter - filter_edge_halfwidth) / float(ori_size)
        if 0 > edge_low:
            edge_low = 0
        edge_high = (ires_filter + filter_edge_halfwidth) / float(ori_size)
        if ft.shape[0] < edge_high:
            edge_high = ft.shape[0]
        edge_width = edge_high - edge_low

        #Generate res array
    
        r = np.fromfunction(resolution_generator, ft.shape, dtype=np.float32, res_sizes = ft.shape)
        r = np.sqrt(r) / ori_size

        low = r > edge_low
        ft[low] *= 0.5 + 0.5 * np.cos(np.pi * (r[low] - edge_low)/edge_width)

        high = r > edge_high
        ft[high] = 0

        ft = np.fft.ifftshift(ft)
        self.data = np.fft.ifft2(ft).real
        self.updateStatistics()

    def highpass_filter(self, angpix, high_pass, filter_edge_width = 2):
        ft = np.fft.fftshift(np.fft.fft2(self.data))
        ori_size = self.nx
        ires_filter = int(round((ori_size * angpix) / high_pass))
        filter_edge_halfwidth = filter_edge_width / 2

        edge_low = (ires_filter - filter_edge_halfwidth) / float(ori_size)
        if 0 > edge_low:
            edge_low = 0
        edge_high = (ires_filter + filter_edge_halfwidth) / float(ori_size)
        if ft.shape[0] < edge_high:
            edge_high = ft.shape[0]
        edge_width = edge_high - edge_low

        #Generate res array

        r = np.fromfunction(resolution_generator, ft.shape, dtype=np.float32, res_sizes = ft.shape)
        r = np.sqrt(r) / ori_size

        high = r < edge_high
        ft[high] *= 0.5 + 0.5 * np.cos(np.pi * (r[high] - edge_high)/edge_width)

        low = r < edge_low
        ft[low] = 0

        ft = np.fft.ifftshift(ft)
        self.data = np.fft.ifft2(ft).real
        self.updateStatistics()

    def apply_gaussian(self, sigma):
        self.data = gaussian_filter(self.data, sigma)
        self.updateStatistics()
    

    def set_ctf_values(self, defocus_u, defocus_v, defocus_angle, voltage, cs, q0, bfac):
        self.defocus_u = defocus_u
        self.defocus_v = defocus_v
        self.defocus_angle = defocus_angle / 360.0 * np.pi * 2
        self.voltage = voltage * 1e3
        self.cs = cs * 1e7
        self.q0 = q0
        self.bfac = bfac
        self.defocus_average = -0.5 * (self.defocus_u + self.defocus_v)
        self.defocus_deviation = -0.5 * (self.defocus_u - self.defocus_v)
        self.lmbda = 12.2643247 / np.sqrt(self.voltage * (1. + self.voltage * 0.978466e-6))
        
        self.K1 = np.pi / 2 * 2 * self.lmbda;
        self.K2 = np.pi / 2 * self.cs * self.lmbda * self.lmbda * self.lmbda;
        self.K3 = np.sqrt(1-self.q0*self.q0);
        self.K4 = -self.bfac / 4.;

    def set_ctf_values_from_dict(self, values):
        self.set_ctf_values(values['defocus_u'], values['defocus_v'], values['defocus_angle'], values['voltage'], values['cs'], values['q0'], 0)

    def calculate_ctf(self, apix):
        xy = np.fromfunction(ctf_xy_generator, (self.nx, self.ny), dtype=np.float32, nx = self.nx, angpix = apix)
        u2 = xy[0] * xy[0] + xy[1] * xy[1]
        u = np.sqrt(u2)
        u4 = u2 * u2

        ellipsoid_ang = np.arctan2(xy[1], xy[0]) - self.defocus_angle
        cos_ellipsoid_ang_2 = np.cos(2 * ellipsoid_ang)
        deltaF = self.defocus_average + self.defocus_deviation * cos_ellipsoid_ang_2
        tooSmall = np.abs(xy[0]) + np.abs(xy[1]) < 2e-6
        deltaF[tooSmall] = 0

        argument = self.K1 * deltaF * u2 + self.K2 * u4
        ctf = -1 * (self.K3 * np.sin(argument) - self.q0 * np.cos(argument))

        E = np.exp(self.K4 * u2)
        ctf *= E

        return ctf

    def ctf_correct(self, apix):
        ctf = self.calculate_ctf(apix)
        ft = np.fft.fftshift(np.fft.fft2(self.data))
        ft *= ctf
        ft = np.fft.ifftshift(ft)
        self.data = np.fft.ifft2(ft).real
        self.updateStatistics()