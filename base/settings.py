# coding=utf-8

# Local Application/Library Specific Imports
from base.utils import load_config, setup_logger
import PySpin

class LoggerMixin:
    """
        A mixin class to add logging functionality to other classes using configuration settings.

        The LoggerMixin class provides a convenient way to add a logger instance to any class that inherits from it.
        The logger instance is set up using configuration settings loaded from a JSON file. The derived class can then
        use the logger to log messages with various severity levels.

        Attributes:
            config (dict): Dictionary containing the configuration settings loaded from the JSON config file.
            logger (logging.Logger): Configured logger instance for logging messages.
    """
    
    def __init__(self):
        """
            Initialize the LoggerMixin instance, loading configuration settings and setting up the logger.
        """
        self.config = load_config()
        self.logger = setup_logger(
            name="my_logger",
            log_dir=self.config["log_output"],
            log_filename=self.config["log_file"],
            max_size=self.config["log_file_size"],
            backup_count=self.config["backup_count"]
        )


class CameraParameters(LoggerMixin):
    """
        This class provides access to the list of functions which can configure the required setting of the camera.
    """

    def __init__(self):
            super().__init__()


    def set_acquisition_mode(self, nodemap, mode='Continuous'):
        """
            This function is for setting the acquisition mode. There are three different modes: Continuous, SingleFrame, MultiFrame.
            
            Parameters:
                nodemap (PySpin.INodeMap): The device nodemap.
                mode (str): The desired acquisition mode, options are 'SingleFrame', 'MultiFrame', 'Continuous'. Default is 'Continuous'.
            
            Returns:
                None
            Raises:
                PySpin.SpinnakerException: If the 'AcquisitionMode' node is not available or not writable.
        """
        # check if the mode is valid
        if mode not in ('SingleFrame', 'MultiFrame', 'Continuous'):
            raise ValueError("Invalid mode. Please provide a valid mode ('SingleFrame', 'MultiFrame', 'Continuous')")
    
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            raise PySpin.SpinnakerException(f'Unable to set acquisition mode to {mode} (enum retrieval). Aborting...')

        # Retrieve entry node from enumeration node
        entry_acquisition_mode = node_acquisition_mode.GetEntryByName(mode)
        if not PySpin.IsAvailable(entry_acquisition_mode) or not PySpin.IsReadable(entry_acquisition_mode):
            raise PySpin.SpinnakerException(f'Unable to set acquisition mode to {mode} (entry retrieval). Aborting...')

        # Retrieve integer value from entry node
        acquisition_mode = entry_acquisition_mode.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_acquisition_mode.SetIntValue(acquisition_mode)

        self.logger.info(f'Acquisition mode set to  {mode}')


    def set_pixel_format(self, nodemap, format='Mono8'):
        """
            This function configures the pixel format for the images. This settings must be applied before 
            BeginAcquisition() is called; otherwise, they will be read only.

            Parameters:
                nodemap (PySpin.INodeMap): The device nodemap.
                format (str): Pixel Format (Mono8, Mono12, Mono16)

            Returns:
                bool: True if pixel format is set successfully, False otherwise.
            
            Raises:
                PySpin.SpinnakerException: If the 'PixelFormat' node is not available or not writable.
        """
        node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
        if not PySpin.IsAvailable(node_pixel_format) or not PySpin.IsWritable(node_pixel_format):
            raise PySpin.SpinnakerException(f'Unable to set the PixelFormat to {format} (enum retrieval). Aborting...')
        
        # Retrieve the desired entry node from the enumeration node
        entry_pixel_format = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName(format))
        if not PySpin.IsAvailable(entry_pixel_format) or not PySpin.IsReadable(entry_pixel_format):
            raise PySpin.SpinnakerException(f'Unable to set the PixelFormat to {format} (entry retrieval). Aborting...')
        # Retrieve the integer value from the entry node
        pixel_format = entry_pixel_format.GetValue()

        # Set integer as new value for enumeration node
        node_pixel_format.SetIntValue(pixel_format)
        self.logger.info(f'PixelFormat set to {format}')


    def set_bufferhandling_mode(self, sNodemap, mode='NewestOnly'):
        """
            This function configures the buffer handling mode for the stream. This setting must be applied before 
            BeginAcquisition() is called; otherwise, it will be read only. 
            There are four different modes: 
                - NewestOnly: Only the newest image is available for access.
                - NewestFirst: Images are stored in a circular buffer and the newest image is always at the front.
                - OldestFirstOverwrite: Images are stored in a circular buffer and the oldest image is overwritten.
                - OldestFirst: Images are stored in a circular buffer and the oldest image is always at the front.

            Parameters:
                cam (PySpin.Camera): Camera instance to grab images from.
                sNodemap (PySpin.INodeMap): The device nodemap.
                mode (str): buffer handling mode. (NewestOnly, NewestFirst, OldestFirstOverwrite, OldestFirst)

            Returns:
                bool: True if buffer handling mode is set successfully, False otherwise.

            Raises:
                PySpin.SpinnakerException: If the 'StreamBufferHandlingMode' node is not available or not writable.
        """
        # Set bufferhandling mode
        node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
        if not PySpin.IsAvailable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
            raise PySpin.SpinnakerException(f'Unable to set stream buffer handling mode to {mode} (enum retrieval). Aborting...')

        # Retrieve entry node from enumeration node
        entry_buffer_mode = node_bufferhandling_mode.GetEntryByName(mode)
        if not PySpin.IsAvailable(entry_buffer_mode) or not PySpin.IsReadable(entry_buffer_mode):
            raise PySpin.SpinnakerException(f'Unable to set stream buffer handling mode to {mode} (entry retrieval). Aborting...')

        # Retrieve integer value from entry node
        get_node_mode = entry_buffer_mode.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_bufferhandling_mode.SetIntValue(get_node_mode)
        self.logger.info(f'Buffer Handling Mode is set to {mode} {get_node_mode}')


    def set_auto_exposure_mode(self, nodemap, mode='Off'):
        """
            This function configures the Auto Exposure function for the camera. This setting can be used to control how the camera's exposure is automatically adjusted. 
            There are three different modes:
                - Off: Exposure is not automatically adjusted.
                - Once: Exposure is automatically adjusted once.
                - Continuous: Exposure is continuously automatically adjusted.
            
            Parameters:
                nodemap (PySpin.INodeMap): The device nodemap.
                mode (str): Auto exposure mode. (Off, Once, Continuous)
            
            Returns:
                bool: True if Auto Exposure mode is set successfully, False otherwise.
            
            Raises:
                PySpin.SpinnakerException: If the 'ExposureAuto' node is not available or not writable.
        """
        if mode not in ['Off', 'Once', 'Continuous']:
            self.logger.info(f'Invalid mode {mode} selected. Please choose one of the following modes: Off, Once, Continuous')
            return False
    
        # Set Auto Exposure mode
        node_exposure_mode = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
        if not PySpin.IsAvailable(node_exposure_mode) or not PySpin.IsWritable(node_exposure_mode):
            raise PySpin.SpinnakerException(f'Unable to set Exposure Auto to {mode} (enum retrieval). Aborting...')

        entry_exposure_mode = node_exposure_mode.GetEntryByName(mode)
        if not PySpin.IsAvailable(entry_exposure_mode) or not PySpin.IsReadable(entry_exposure_mode):
            raise PySpin.SpinnakerException(f'Unable to set Exposure Auto to {mode} (entry retrieval). Aborting...')

        exposure_auto_mode = entry_exposure_mode.GetValue()

        node_exposure_mode.SetIntValue(exposure_auto_mode)

        self.logger.info(f'Auto Exposure mode set to {mode} ')
        return True


    def set_exposure(self, nodemap, exp_seconds):
        """
            Configures the camera to set a manual exposure time value.

            Parameters:
                nodemap (PySpin.INodeMap): The device nodemap.
                exp_seconds (float): The desired exposure time in microseconds. If a value less than the minimum or greater than the maximum supported by the camera is passed, an error message is printed and the function returns False.
            Returns:
                bool: True if the exposure time is set successfully, False otherwise.
            Raises:
                PySpin.SpinnakerException: If the 'ExposureTime' node is not available or not writable.
        """
        self.set_auto_exposure_mode(nodemap, mode='Off')
        
        # Get minimum and maximum exposure time values supported by the camera
        node_exp_time_min = PySpin.CFloatPtr(nodemap.GetNode('ExposureTimeAbs'))
        min_exp_time = node_exp_time_min.GetMin()
        node_exp_time_max = PySpin.CFloatPtr(nodemap.GetNode('ExposureTimeAbs'))
        max_exp_time = node_exp_time_max.GetMax()
        
        if exp_seconds < min_exp_time or exp_seconds > max_exp_time:
            self.logger.info(f'Exposure time of {exp_seconds} microseconds is not within the supported range of the camera. The supported range is between {min_exp_time} and {max_exp_time} microseconds.')
            return False

        # Set Exposure Time to less than 1/50th of a second (5000 micro second is used as an example)
        node_exposure_time = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
        if not PySpin.IsAvailable(node_exposure_time) or not PySpin.IsWritable(node_exposure_time):
            raise PySpin.SpinnakerException(f'\nUnable to set Exposure Time (float retrieval). Aborting...\n')

        node_exposure_time.SetValue(exp_seconds)
        self.logger.info(f'Exposure Set to {exp_seconds}')
        return True


    def set_auto_gain(self, nodemap, mode='Off'):
        """
            This function configures the camera's auto gain feature.
            There are three different modes: 
                - Off: The gain is set manually.
                - Once: The gain is set automatically once.
                - Continuous: The gain is set automatically continuously.

            :param nodemap: Device nodemap.
            :param mode: 'Off', "Once', 'Continuous'
            :return: True if the auto gain mode is set successfully, False otherwise.
        """
        if mode not in ['Off', 'Once', 'Continuous']:
            self.logger.info(f'Invalid mode {mode} selected. Please choose one of the following modes: Off, Once, Continuous')
            return False
        
        node_gain_mode = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
        if not PySpin.IsAvailable(node_gain_mode) or not PySpin.IsWritable(node_gain_mode):
            raise PySpin.SpinnakerException(f'Unable to set Gain Auto to {mode} (enum retrieval). Aborting...')

        entry_gain_mode = node_gain_mode.GetEntryByName(mode)
        if not PySpin.IsAvailable(entry_gain_mode) or not PySpin.IsReadable(entry_gain_mode):
            raise PySpin.SpinnakerException(f'Unable to set Gain Auto to {mode} (entry retrieval). Aborting...')

        gain_auto_mode = entry_gain_mode.GetValue()

        node_gain_mode.SetIntValue(gain_auto_mode)

        self.logger.info(f'Auto Gain mode set to {mode} ')
        return True


    def set_gain(self, nodemap, gain_val: int):
        """
            Configures the camera to set a manual gain value.
            
            Parameters:
                nodemap (PySpin.INodeMap): Device nodemap.
                gain_val (int): Gain value in integer.
            
            Returns:
                bool: True if gain value is set successfully, False otherwise.
        """
        # Get minimum and maximum gain values supported by the camera
        node_gain_min = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
        min_gain = node_gain_min.GetMin()
        node_gain_max = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
        max_gain = node_gain_max.GetMax()
        
        if gain_val < min_gain or gain_val > max_gain:
            raise PySpin.SpinnakerException(f'Gain value of {gain_val} is not within the supported range of the camera. The supported range is between {min_gain} and {max_gain}.')

        # Turn off auto Gain
        self.set_auto_gain(nodemap, mode='Off')

        # Set Gain
        node_gain_val = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
        if not PySpin.IsAvailable(node_gain_val) or not PySpin.IsWritable(node_gain_val):
            raise PySpin.SpinnakerException('\nUnable to set Gain (float retrieval). Aborting...\n')

        node_gain_val.SetValue(gain_val)
        self.logger.info(f'Gain Value is set to {gain_val}')
        return True


    def enable_framerate(self, nodemap, enable: bool):
        """
            Enable or disable the framerate of a camera using the PySpin library.
            
            Parameters:
                nodemap (PySpin.INodeMap): The device nodemap.
                enable (bool): A boolean value that indicates whether to enable or disable the framerate.
                
            Returns:
                bool: True if the framerate was set successfully, False otherwise.
            Raises:
                PySpin.SpinnakerException: If the 'AcquisitionFrameRateEnable' node is not available or not writable.
        """
        enable_fps = PySpin.CBooleanPtr(nodemap.GetNode('AcquisitionFrameRateEnable'))
        if not PySpin.IsAvailable(enable_fps) or not PySpin.IsWritable(enable_fps):
            raise PySpin.SpinnakerException('Unable to change the value for framerate. The AcquisitionFrameRateEnable node is not available or not writable.')
        enable_fps.SetValue(enable)


    def acquisition_framerate(self, nodemap):
        """
            Displays the acquisition frame rate of a camera using the PySpin library.
            
            Parameters:
                nodemap (PySpin.INodeMap): The device nodemap.
                
            Returns:
                float: The current Acquisition frame rate.
            Raises:
                PySpin.SpinnakerException: If the 'AcquisitionFrameRate' node is not available or not readable
        """
        node_acquisition_fps = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
        if not PySpin.IsAvailable(node_acquisition_fps) or not PySpin.IsReadable(node_acquisition_fps):
            raise PySpin.SpinnakerException('Unable to Get Acquisition Frame Rate. The AcquisitionFrameRate node is not available or not readable.')
        return node_acquisition_fps.GetValue()


    def get_resulting_framerate(self, nodemap):
        """
            Gets the camera's resulting frame rate.

            Parameters:
                nodemap (PySpin.INodeMap): The camera's nodemap.

            Returns:
                float: The resulting frame rate value, or None if not available.
        """
        try:
            # Check if the AcquisitionResultingFrameRate node is available
            resulting_framerate_node = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionResultingFrameRate'))

            if PySpin.IsAvailable(resulting_framerate_node) and PySpin.IsReadable(resulting_framerate_node):
                resulting_framerate = resulting_framerate_node.GetValue()
                self.logger.info(f"Resulting frame rate: {resulting_framerate} fps.")
                return resulting_framerate
            else:
                self.logger.error("Unable to get resulting frame rate. The AcquisitionResultingFrameRate node is not available or not readable.")
                return None
        except PySpin.SpinnakerException as ex:
            self.logger.error(f"Error: {ex}")
            return None


    def set_gamma(self, nodemap, gamma_val: float):
        """
            Sets the gamma value for the camera. 
            
            Parameters:
                nodemap (PySpin.INodeMap): The device nodemap.
                gamma_val (float): Gamma value to set the camera to.
                
            Returns:
                float: The current gamma value.
            Raises:
                PySpin.SpinnakerException: If the 'Gamma' node is not available or not writable.
        """
        if  not (gamma_val < 3.9):
            raise ValueError(f"Gamma Value shoud be less than 3.9")
        node_gamma = PySpin.CFloatPtr(nodemap.GetNode('Gamma'))
        if not PySpin.IsAvailable(node_gamma) or not PySpin.IsWritable(node_gamma):
            raise PySpin.SpinnakerException('Unable to set Gamma value. The Gamma node is not available or not writable.')
        node_gamma.SetValue(gamma_val)
        self.logger.info(f'Gamma Value is set to {gamma_val}')
        return node_gamma.GetValue()


    def auto_sharp(self, nodemap, enable: bool):
        """
            Enables or disables the auto sharpening feature for a camera. 
            
            Parameters:
                nodemap (PySpin.INodeMap): The device nodemap.
                enable (bool): True to enable, False to disable.
                
            Returns:
                bool: The current value of the 'SharpeningAuto' node.
            Raises:
                PySpin.SpinnakerException: If the 'SharpeningAuto' node is not available or not writable.
        """
        node_sharp = PySpin.CBooleanPtr(nodemap.GetNode('SharpeningAuto'))
        if not PySpin.IsAvailable(node_sharp) or not PySpin.IsWritable(node_sharp):
            raise PySpin.SpinnakerException('Unable to change the value for auto sharpening feature. The SharpeningAuto node is not available or not writable.')
        node_sharp.SetValue(enable)
        return node_sharp.GetValue()


    def enable_sharpening(self, nodemap, enable: bool):
        """
            Enables or disables the sharpening feature for a camera. 
            
            Parameters:
                nodemap (PySpin.INodeMap): The device nodemap.
                enable (bool): True to enable, False to disable.
                
            Returns:
                bool: The current value of the 'SharpeningEnable' node.
            Raises:
                PySpin.SpinnakerException: If the 'SharpeningEnable' node is not available or not writable.
        """
        node_sharp_enable = PySpin.CBooleanPtr(nodemap.GetNode('SharpeningEnable'))
        if not PySpin.IsAvailable(node_sharp_enable) or not PySpin.IsWritable(node_sharp_enable):
            raise PySpin.SpinnakerException('Unable to change the value for sharpening feature. The SharpeningEnable node is not available or not writable.')
        node_sharp_enable.SetValue(enable)
        return node_sharp_enable.GetValue()


    def set_sharpening(self, nodemap, sharp_val: float):
        """
            Changes the sharpening value for a camera. 
            
            Parameters:
                nodemap (PySpin.INodeMap): The device nodemap.
                sharp_val (int): Sharpening value between 1 to 8.
                
            Returns:
                float: The current value of the 'Sharpening' node.
            Raises:
                PySpin.SpinnakerException: If the 'Sharpening' node is not available or not writable.
                ValueError: If the sharp_val is not between 1 and 8.
        """
        if not (1 <= sharp_val <= 8):
            raise ValueError("sharp_val should be between 1 to 8")
        node_sharp_val = PySpin.CFloatPtr(nodemap.GetNode('Sharpening'))
        if not PySpin.IsAvailable(node_sharp_val) or not PySpin.IsWritable(node_sharp_val):
            raise PySpin.SpinnakerException('Unable to set Sharpening. The Sharpening node is not available or not writable.')
        node_sharp_val.SetValue(sharp_val)
        return node_sharp_val.GetValue()


    def auto_white_balance(self, nodemap, mode='Off'):
        """
            Changes the Auto White Balance mode for a camera.
            
            Parameters:
                nodemap (PySpin.INodeMap): The device nodemap.
                mode (str): Auto White Balance mode options: 'Off', 'Once', 'Continuous'. Default value is 'Off'.
            
            Raises:
                PySpin.SpinnakerException: If the 'BalanceWhiteAuto' node is not available or not writable.
                ValueError: If the mode is not one of the options: 'Off', 'Once', 'Continuous'
        """
        if mode not in ['Off', 'Once', 'Continuous']:
            raise ValueError("mode should be one of the options: 'Off', 'Once', 'Continuous'")
        node_whitebalance_mode = PySpin.CEnumerationPtr(nodemap.GetNode('BalanceWhiteAuto'))
        
        if not PySpin.IsAvailable(node_whitebalance_mode) or not PySpin.IsWritable(node_whitebalance_mode):
            raise PySpin.SpinnakerException(f'Unable to set BalanceWhiteAuto to {mode} (enumeration retrieval).')

        entry_whitebalance_mode = node_whitebalance_mode.GetEntryByName(mode)
        if not PySpin.IsAvailable(entry_whitebalance_mode) or not PySpin.IsReadable(entry_whitebalance_mode):
            raise PySpin.SpinnakerException(f'Unable to set BalanceWhiteAuto to {mode} (entry retrieval).')

        white_balance_mode = entry_whitebalance_mode.GetValue()
        node_whitebalance_mode.SetIntValue(white_balance_mode)
        self.logger.info(f'Balance White Auto mode set to {mode}')  


    def auto_blacklevel(self, nodemap, mode='Off'):
        """
            Changes the black level mode of a camera using the PySpin library.
    
            Parameters:
                nodemap (PySpin.INodeMap): The device nodemap.
                mode (str): The desired black level mode. Can be 'Off', 'Once' or 'Continuous' (default 'Off').
                
            Returns:
                None
            Raises:
                ValueError: If the mode passed is not valid ('Off', 'Once' or 'Continuous')
                PySpin.SpinnakerException: If the 'BlackLevelAuto' node is not available or not writable.
        """
        node_autoblack_mode = PySpin.CEnumerationPtr(nodemap.GetNode('BlackLevelAuto'))
        if not PySpin.IsAvailable(node_autoblack_mode) or not PySpin.IsWritable(node_autoblack_mode):
            self.logger.warning('\nUnable to set Auto Blacklevel mode. Aborting...\n')
            return False

        entry_autoblack_mode = node_autoblack_mode.GetEntryByName(mode)
        if not PySpin.IsAvailable(entry_autoblack_mode) or not PySpin.IsReadable(entry_autoblack_mode):
            self.logger.warning('\nUnable to set Auto Blacklevel Mode. Aborting...\n')
            return False

        autoblack_mode = entry_autoblack_mode.GetValue()

        node_autoblack_mode.SetIntValue(autoblack_mode)
        self.logger.info(f'AutoBlack is set to {mode} mode')


    def set_blacklevel(self, nodemap, black_val: float):
        """
            Sets the black level value of a camera using the PySpin library.
            
            Parameters:
                nodemap (PySpin.INodeMap): The device nodemap.
                black_val (float): The desired black level value in percent.
                
            Raises:
                PySpin.SpinnakerException: If the 'BlackLevel' node is not available or not writable.
        """
        node_black_level = PySpin.CFloatPtr(nodemap.GetNode('BlackLevel'))
        if not PySpin.IsAvailable(node_black_level) or not PySpin.IsWritable(node_black_level):
            raise PySpin.SpinnakerException('Unable to set BlackLevel. The BlackLevel node is not available or not writable.')

        node_black_level.SetValue(black_val)
        self.logger.info(f'BlackLevel Set to {black_val}')


class DeviceInfo(LoggerMixin):
    """
        This class provides access to the functions which retrieve details information about the devices.
    """
    
    def __init__(self):
            super().__init__()


    def get_device_info(self, nodemap):
        """
            This function retrieves the device information of the camera from the transport layer
            and logs the information without storing it in a dictionary.

            Parameters:
                nodemap: Device nodemap.
            Returns:
                return: True if the information is logged successfully, False otherwise.
        """
        try:
            node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

            if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
                features = node_device_information.GetFeatures()
                for feature in features:
                    node_feature = PySpin.CValuePtr(feature)
                    if PySpin.IsReadable(node_feature):
                        feature_name = node_feature.GetName()
                        feature_value = node_feature.ToString()
                        self.logger.info(f'{feature_name}: {feature_value}')
                    else:
                        self.logger.info(f'{node_feature.GetName()} is not readable')

        except PySpin.SpinnakerException as ex:
            self.logger.error(f'Error: {ex}')
            return False

        return True


class BufferHandlingControl(LoggerMixin):
    """
        This class provides access to handle the buffering process for Image acquisition.
    """
    
    def __init__(self):
        super().__init__()


    def buffer_count_mode(self, nodemap, mode='Manual'):
        """
            This function sets the buffer count mode for an image acquisition using the Spinnaker SDK.
            
            Parameters:
                nodemap (PySpin.INodeMap): a PySpin NodeMap object, which contains all the nodes of the camera.
                mode (str): a string representing the buffer count mode. The default value is 'Manual', 
                            which means that the buffer count is set manually.
            
            Returns:
                bool: True if the buffer count mode is set successfully, False otherwise.
        """
        try:
            # Retrieve the node for the StreamBufferCountMode
            node_buffer_mode = PySpin.CEnumerationPtr(nodemap.GetNode('StreamBufferCountMode'))
            
            # Check if the node is available and writable
            if not PySpin.IsAvailable(node_buffer_mode) or not PySpin.IsWritable(node_buffer_mode):
                raise PySpin.SpinnakerException(f'Unable to set Stream Buffer Count Mode to {mode} (enum retrieval). Aborting...')

            # Retrieve the entry for the desired buffer count mode
            entry_buffer_mode = node_buffer_mode.GetEntryByName(mode)
            
            # Check if the entry is available and readable
            if not PySpin.IsAvailable(entry_buffer_mode) or not PySpin.IsReadable(entry_buffer_mode):
                raise PySpin.SpinnakerException(f'Unable to set Stream Buffer Count Mode to {mode} (entry retrieval). Aborting...')

            # Get the value of the buffer count mode
            buffer_count_mode = entry_buffer_mode.GetValue()

            # Set the buffer count mode
            node_buffer_mode.SetIntValue(buffer_count_mode)

            print(f'Stream Buffer Count mode set to {mode} ')
            return True
        except PySpin.SpinnakerException as ex:
            self.logger.error(f'Error: {ex}')
            return False


    def set_buffer_count(self, sNodemap, num_buffer: int):
        """
            This function sets the buffer count for an image acquisition using the Spinnaker SDK.
            
            Parameters:
                sNodemap (PySpin.INodeMap): a PySpin Stream NodeMap object, which contains all the nodes of the camera.
                num_buffer (int): an integer representing the number of images to be buffered.
            
            Returns:
                bool: True if the buffer count is set successfully, False otherwise.
        """
        try:
            # Retrieve the node for the StreamBufferCountManual
            buffer_count = PySpin.CIntegerPtr(sNodemap.GetNode('StreamBufferCountManual'))
            
            # Check if the node is available and writable
            if not PySpin.IsAvailable(buffer_count) or not PySpin.IsWritable(buffer_count):
                raise PySpin.SpinnakerException('Unable to set Buffer Count (Integer node retrieval). Aborting...\n')

            self.logger.info(f'Default Buffer Count: {buffer_count.GetValue()}')
            self.logger.info(f'Maximum Buffer Count: {buffer_count.GetMax()}')

            # Set the buffer count to the desired value
            buffer_count.SetValue(num_buffer)
            self.logger.info(f'Buffer count now set to: {buffer_count.GetValue()}')
            return True
        
        except PySpin.SpinnakerException as ex:
            self.logger.error(f'Error: {ex}')
            return False

