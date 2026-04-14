
# Standard Library Imports
import gc
import timeit
from contextlib import contextmanager
from enum import Enum

# Third Party Imports
import torch

# Local Application/Library Specific Imports
from base.acquisition import Acquisition
from base.settings import CameraParameters, DeviceInfo, LoggerMixin
from base.data_processing import PostDataProcessor
import PySpin


class MainApp(LoggerMixin):
    
    class Action(Enum):
        PREVIEWING = 'Previewing'
        PROCESSING = 'Processing'
        H5PROCESSING = 'H5Processing'
        SAVING = 'Saving'
        CAPTURING = 'Capturing'
        UPDATE = 'Update'


    def __init__(self):
        super().__init__()
        self.running = False
        self.acquisition = None


    @contextmanager
    def init_camera(self, cam: object):
        cam.Init()
        try:
            yield cam
        finally:
            cam.DeInit()


    def setup_camera(self, cam: object, config: dict):
        nodemap = cam.GetNodeMap()
        sNodemap = cam.GetTLStreamNodeMap()
        nodemap_tldevice = cam.GetTLDeviceNodeMap()

        device_info = DeviceInfo()
        device_info.get_device_info(nodemap_tldevice)

        camera_parameters = CameraParameters()
        camera_parameters.set_pixel_format(nodemap, format=config["pixel_format"])
        camera_parameters.set_exposure(nodemap, exp_seconds=config["exposure_seconds"])
        camera_parameters.set_bufferhandling_mode(sNodemap, mode=config["buffer_handling_mode"])
        camera_parameters.set_acquisition_mode(nodemap, mode=config["acquisition_mode"])
        camera_parameters.set_gain(nodemap, gain_val=config["gain_value"])
        camera_parameters.set_gamma(nodemap, gamma_val=config["gamma_value"])
        camera_parameters.set_blacklevel(nodemap, black_val=config["blacklevel_value"])


    def perform_action(self, cam: object, action: Action, config: dict):
        # if self.acquisition is None:
        #     self.acquisition = Acquisition(self, config)
        self.acquisition = Acquisition(self, config)
        cam.BeginAcquisition()
        try:
            if action == MainApp.Action.PREVIEWING:
                self.acquisition.preview_images(cam)
            elif action == MainApp.Action.PROCESSING:
                self.acquisition.process_buffer_images(cam, num_buffer=config["num_buffer"])
                self.acquisition.process_images(save_images=config["save_images"], save_predicted_images=config["save_predicted_images"], cropping_cells=config["cropping_cells"], output_dir=config["img_output"])
                torch.cuda.empty_cache()
            # elif action == MainApp.Action.H5PROCESSING:
            #     self.acquisition.load_h5file(h5_file_path=h5_file_path)
            #     self.acquisition.process_images(save_images=config["save_images"], save_predicted_images=config["save_predicted_images"], cropping_cells=config["cropping_cells"], output_dir=config["img_output"])
            elif action == MainApp.Action.SAVING:
                self.acquisition.save_buffer_images(cam, num_buffer=config["num_buffer"], image_type=config["image_type"])
                self.logger.info("Saving Hologram Images Complete")
            elif action == MainApp.Action.UPDATE:
                self.acquisition.display_images_queue.get()
        finally:
            cam.EndAcquisition()


    def process_h5_file(self, h5_file_path: str, ovizio_reconstruction: bool, ai_reconstruction: bool, png_upload: bool, post_data_processing: bool, config: dict):
        self.logger.info("Process is started.")
        try:
            if self.acquisition is None:
                self.acquisition = Acquisition(self, config)
            if ovizio_reconstruction and png_upload==False:
                file_list = h5_file_path.split(',')
                for img_path in file_list:
                    self.logger.info(f'Ovizio Reconstruction Started.')
                    reconst_start_time = timeit.default_timer()
                    self.acquisition.reconstruct_images(save_phase_images=False, save_amp_images=False, ovizio_reconstruction=ovizio_reconstruction, h5_file_path=img_path)
                    reconst_duration = timeit.default_timer() - reconst_start_time
                    self.logger.info(f'Total Recnostruction Time (Ovizio): {reconst_duration}')
                    detect_start_time = timeit.default_timer()
                    self.acquisition.process_data(img_path=img_path, ovizio_reconstruction=ovizio_reconstruction, png_upload=png_upload)
                    detect_duration = timeit.default_timer() - detect_start_time
                    self.logger.info(f'Detection + Post Processing Time (Ovizio): {detect_duration} seconds')
                    torch.cuda.empty_cache()
                    gc.collect()
            elif ovizio_reconstruction==False and png_upload==True:
                file_list = h5_file_path.split(',')
                for img_path in file_list:
                    detect_start_time = timeit.default_timer()
                    self.acquisition.process_data(img_path=img_path, ovizio_reconstruction=ovizio_reconstruction, png_upload=png_upload)
                    detect_duration = timeit.default_timer() - detect_start_time
                    self.logger.info(f'The Complete Processing Time (Ovizio): {detect_duration} seconds')
                    torch.cuda.empty_cache()
                    gc.collect()
            elif ovizio_reconstruction==False and ai_reconstruction==True:
                if self.config["parallel_processing"]:
                    file_list = h5_file_path.split(',')
                    for img_path in file_list:
                        self.logger.info({img_path})
                        self.acquisition.run_parallel_reconstruction_and_detection(img_path)
                else:
                    file_list = h5_file_path.split(',')
                    for img_path in file_list:
                        self.acquisition.load_h5file(h5_file_path=img_path)
                        self.logger.info(f'AI Reconstruction Started.')
                        reconst_start_time = timeit.default_timer()
                        self.acquisition.reconstruct_images(save_phase_images=self.config["save_phase_images"], save_amp_images=self.config["save_amp_images"], ovizio_reconstruction=ovizio_reconstruction, h5_file_path=img_path)
                        reconst_duration = timeit.default_timer() - reconst_start_time
                        self.logger.info(f'Total AI Reconstruction Time: {reconst_duration}')
                        detect_start_time = timeit.default_timer()
                        self.acquisition.process_data(img_path=img_path, ovizio_reconstruction=ovizio_reconstruction, png_upload=png_upload)
                        detect_duration = timeit.default_timer() - detect_start_time
                        self.logger.info(f'Detection + Post Processing Time (AI): {detect_duration} seconds')
                        torch.cuda.empty_cache()
                        gc.collect()
            elif post_data_processing==True:
                self.logger.info("Post Data Processing.")
                data_processor = PostDataProcessor()
                data_processor.load_csv_file(csv_file_path=h5_file_path)
                self.logger.info("Post Data Processing Finished.")
        finally:
                torch.cuda.empty_cache()
                gc.collect()


    def process_camera_images(self, cam: object, config: dict, action: Action):
        try:
            with self.init_camera(cam):
                self.setup_camera(cam, config)
                for result in self.perform_action(cam, action, config):
                    yield result
        except PySpin.SpinnakerException as ex:
            self.logger.error(f'Error: {ex}')
        return


    # def run(self, action: str, h5_file_path: str):
    def run(self, action: str):
        self.running = True
        action = MainApp.Action(action)
        config = self.config
        self.logger.info('Acquisition Started.')
        
        try:
            # Retrieve singleton reference to system object
            system = PySpin.System.GetInstance()
            # Get current library version
            version = system.GetLibraryVersion()
            self.logger.info(f'Library version: {version.major}.{version.minor}.{version.type}.{version.build}')
            # Retrieve list of cameras from the system
            cam_list = system.GetCameras()
            num_cameras = cam_list.GetSize()
            self.logger.info(f'Number of cameras detected: {num_cameras}')
            # Finish if there are no cameras
            if num_cameras == 0:
                self.logger.error('Not enough cameras!')
                yield None
            for i, cam in enumerate(cam_list):
                # Assuming setup_camera and perform_action are defined elsewhere
                with self.init_camera(cam):
                    self.setup_camera(cam, self.config)
                    while self.running: 
                        self.perform_action(cam, action, self.config)
                        if not self.running:  # Check the flag again after each action
                            break
                del cam
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            yield None
        finally:
            # Clear camera list before releasing system
            cam_list.Clear()
            # Release system instance
            system.ReleaseInstance()            

