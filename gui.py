import PySimpleGUI as sg
import os

import neural_style_gui
import utility

from datetime import datetime

def main():

    layout = [
            [sg.Text('Input for Style', size=(15, 1)), sg.Input(key='style_image'), sg.FileBrowse()],
            [sg.Text('Content Image', size=(15, 1)), sg.Input(key='content_image'), sg.FileBrowse()],
            [sg.Text('Output Directory', size=(15, 1)), sg.InputText(key='output_image'), sg.FolderBrowse()],
            [sg.Text('Output Size', size=(15, 1)), sg.InputText('512', key='image_size')],

            [sg.Text('Style Blend Weights', size=(15, 1))],
            [sg.Text('Input', size=(15, 1)), sg.Slider(range=(1, 100), orientation='h', size=(34, 20), default_value=50), sg.Text('Output', size=(15, 1))],

            [sg.Text('Selected GPU', size=(15, 1)), sg.InputText('0', key='gpu')],
            [sg.Text('Iteractions', size=(15, 1)), sg.InputText('1000', key='num_iterations')],

            [sg.Submit(key='start'), sg.Cancel(key='quit')]
             ]

    window = sg.Window('Neural Style Transfer',
            layout)

    while (True):

        # This is the code that reads and updates your window
        event, values = window.Read(timeout=100)

        if event == 'Exit' or event is None:
            break

        if event == 'quit':
            break

        if event == 'start':

            # Set output filename
            single_datestring = datetime.strftime(datetime.now(), '%Y-%m-%d_%H.%M.%S')
            photo_path = '%s/%s_%s_single.png' % ( values['output_image'], 'output', single_datestring)

            # Resize style image
            values['style_image'] = utility.openSingleAndCheck(values['style_image'], 'temp/style_image.png', values['image_size'])

            # Resize Content image
            values['content_image'] = utility.openSingleAndCheck(values['content_image'], 'temp/content_image.png', values['image_size'])

            args = {
                'style_image': values['style_image'],
                'content_image': values['content_image'],
                'output_image': photo_path,
                'gpu': int(values['gpu']),

                # Basic options
                'style_blend_weights': None,
                'image_size': int(values['image_size']),

                # Optimization options
                'content_weight': 5e0,
                'style_weight': 1e2,
                'tv_weight': 1e-3,
                'num_iterations': int(values['num_iterations']),
                'init': 'random',
                'init_image': None,
                'optimizer': 'lbfgs',
                'learning_rate': 1e0,
                'lbfgs_num_correction': 100,

                # Output options
                'print_iter': 50,
                'save_iter': 100,

                # Other options
                'style_scale': 1.0,
                'original_colors': 0,
                'pooling': 'max',
                'model_file': 'models/vgg19-d01eb7cb.pth',
                'disable_check': 'store_true',
                'backend': 'nn',
                'cudnn_autotune': 'store_true',
                'seed': -1,

                'content_layers': 'relu4_2',
                'style_layers': 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1',

                'multidevice_strategy': '4,7,29',
            }

            # print(args)

            args = utility.dotdict(args)
            neural_style_gui.main(args)

            sg.Popup('Title', 'The results of the window.')


    window.Close()   # Don't forget to close your window!

if __name__ == '__main__':
    main()
