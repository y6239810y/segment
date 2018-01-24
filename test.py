# -*- coding: UTF-8 -*-
# import SimpleITK as sitk
# import matplotlib.pyplot as plt
# import numpy as np
# def set_Window(image, max, min):
#     array = sitk.GetArrayFromImage(image)
#     array_max = np.max(array)
#     array_min = np.min(array)
#     image_out = sitk.IntensityWindowing(image, array_min * 1.0, array_max * 1.0, min, max)
#     return image_out
#
# reader = sitk.ImageFileReader()
# dicoms = reader.SetFileName('E:/train_data/multi_task_data_train/DONG_QIN_DA/original1/1.2.392.200036.9116.2.5.1.48.1221418331.1492151145.151352.dcm')
# image = reader.Execute()
#
#
#
# blurFilter = sitk.CurvatureFlowImageFilter()
# blurFilter.SetNumberOfIterations( 5 )
# blurFilter.SetTimeStep( 0.125 )
# image = blurFilter.Execute(image)
#
# segmentationFilter = sitk.ConnectedThresholdImageFilter()
# segmentationFilter.SetLower(-1024.0)
# segmentationFilter.SetUpper(-900.0)
#
#
# seed = [5,5,5]
#
# segmentationFilter.AddSeed(seed)
#
# image = segmentationFilter.Execute(image)
#
# writer = sitk.ImageFileWriter()
# writer.SetFileName( 'out.vtk' )
# writer.Execute( image )
#
# image = segmentationFilter.Execute( image )
# image[seed] = 255
#
# writer = sitk.ImageFileWriter()
# writer.SetFileName('out.vtk')
# writer.Execute(image)
from vtk import *

filename = "liver.stl"

reader = vtk.vtkSTLReader()
reader.SetFileName(filename)

polydata = reader.GetOutput()

# Setup actor and mapper
mapper = vtk.vtkPolyDataMapper()
if vtk.VTK_MAJOR_VERSION <= 5:

    mapper.SetInput(polydata)
else:
    print(vtk.VTK_MAJOR_VERSION)
    mapper.SetInputData(polydata)

actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Setup render window, renderer, and interactor
renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)
renderer.AddActor(actor)
renderWindow.Render()
renderWindowInteractor.Start()