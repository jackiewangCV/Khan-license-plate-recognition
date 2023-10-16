import sys

import gi
gi.require_version('Gst', '1.0')
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GLib, Gst, GstRtspServer

import pyds
import cv2
import numpy as np


class MultiCameraPeopleDetector:
	"""
	This class taken in the list of cameras that has camera addresses.
	For each camera, it creates a source bin and links it to the streammux.
	Then it creates the nvinfer element with specified config file, 
	in this case is for people detection and links it to the streammux. 
	After the inference, the output is converted to RGBA format and sent to the
	application sink. The application sink then converts the buffer to numpy array and 
	puts it in the camera's image queue. The camera's image queue is then retrieved by 
	the web application to stream the video.	
	
	Please let us know if you have any better ideas to do this, especially the streaming part. 
	We are open to suggestions. 
	"""
	def __init__(self, cameras) -> None:
		self.running = False
		self.bitrate = 4000000
		self.pipeline = Gst.Pipeline()

		streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
		if not streammux:
			sys.stderr.write(" Unable to create NvStreamMux \n")
		self.pipeline.add(streammux)		

		
		for i, camera in enumerate(cameras):
			print ("Creating source for", camera.internal_id)
			source_bin = self.create_source_bin(i, camera.cam_address)

			if not source_bin:
				sys.stderr.write("Unable to create source bin \n")

			self.pipeline.add(source_bin)

			padname = "sink_%u" % i
			sinkpad = streammux.get_request_pad(padname)
			if not sinkpad:
				sys.stderr.write("Unable to create sink pad bin \n")

			srcpad = source_bin.get_static_pad("src")
			if not srcpad:
				sys.stderr.write("Unable to create src pad bin \n")
			srcpad.link(sinkpad)

		pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
		if not pgie:
			sys.stderr.write(" Unable to create pgie \n")

		nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
		if not nvvidconv1:
			sys.stderr.write(" Unable to create nvvidconv \n")


		### APP SINK

		capsapp = Gst.ElementFactory.make("capsfilter", "capsapp")
		capsapp.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))

		appsink = Gst.ElementFactory.make("appsink", "appsink")

		if not appsink:
			sys.stderr.write(" Unable to create appsink \n")

		appsink.set_property("emit-signals", True)
		appsink.set_property("sync", False)
		appsink.set_property("max-buffers", 35) ## Pseudo control of the framerate
		appsink.set_property("drop", True)
		appsink.connect("new-sample", self.on_new_sample, appsink, cameras)

		## Streammux settings
		streammux.set_property('live-source', 1)
		streammux.set_property("width", 1280)
		streammux.set_property("height", 720)
		streammux.set_property("batch-size", len(cameras))
		streammux.set_property("batched-push-timeout", 20000) ## Ideally, set it to latency of teh fastest stream

		## PGIE settings
		pgie.set_property("config-file-path", "configs/peoplenet_detector_config.txt")
		pgie.set_property("batch-size", len(cameras))

		mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
		streammux.set_property("nvbuf-memory-type", mem_type)
		nvvidconv1.set_property("nvbuf-memory-type", mem_type)

		self.pipeline.add(pgie)
		self.pipeline.add(nvvidconv1)
		self.pipeline.add(capsapp)
		self.pipeline.add(appsink)

		streammux.link(pgie)
		pgie.link(nvvidconv1)
		nvvidconv1.link(capsapp)
		capsapp.link(appsink)

		self.bus = self.pipeline.get_bus()
		self.bus.add_signal_watch()
		self.bus.connect("message", self.bus_call)

	def on_new_sample(self, sink, appsink, cameras):
		# Get the GstSample from appsink, convert it to GstBuffer, then to numpy array
        # and put it in the dequeue of the camera which will be retrieved from the web application
        # to stream

		sample = sink.emit("pull-sample")
		gst_buffer = sample.get_buffer()
		if not gst_buffer:
			print("Unable to get GstBuffer ")
			return

		# Retrieve batch metadata from the gst_buffer
		# Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
		# C address of gst_buffer as input, which is obtained with hash(gst_buffer)
		batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

		l_frame = batch_meta.frame_meta_list

		while l_frame is not None:
			try:
				frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
			except StopIteration:
				break

			l_obj = frame_meta.obj_meta_list

            # get the numpy array buffer from the appsink
			n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
			n_frame = cv2.cvtColor(n_frame, cv2.COLOR_RGBA2BGRA)

			obj_positions = []

			while l_obj is not None:
				try:
				# Casting l_obj.data to pyds.NvDsObjectMeta
					obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
				except StopIteration:
					break

				if obj_meta.class_id == 0:
					n_frame, pos = self.draw_bounding_boxes(n_frame, obj_meta)
					obj_positions.append(pos)			

				try:
					l_obj = l_obj.next
				except StopIteration:
					break
			
            # Update the camera's image queue with the latest frame
            # and the position queue with the latest positions
            # and the count queue with the latest count

			cameras[frame_meta.source_id].imageq.append(n_frame)
			cameras[frame_meta.source_id].positionq.append(obj_positions)
			cameras[frame_meta.source_id].countq.append(len(obj_positions))

			# print (cameras[frame_meta.source_id].internal_id)

			try:
				l_frame = l_frame.next
			except StopIteration:
				break

		return Gst.FlowReturn.OK


	def draw_bounding_boxes(self, image, obj_meta):
		rect_params = obj_meta.rect_params
		top = int(rect_params.top)
		left = int(rect_params.left)
		width = int(rect_params.width)
		height = int(rect_params.height)

		w_percents = int(width * 0.05) if width > 100 else int(width * 0.1)
		linetop_c1 = (left + w_percents, top)
		linebot_c2 = (left + width - w_percents, top + height)
		image = cv2.rectangle(image, linetop_c1, linebot_c2, (224, 235, 221), 3)

		x1, y1 = linetop_c1
		x2, y2 = linebot_c2
		w = x2 - x1
		h = y2 - y1

		sub_img = image[y1 : y1 + h, x1 : x1 + w]
		white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255

		res = cv2.addWeighted(sub_img, 0.6, white_rect, 0.4, 0.0)

		try:
			image[y1 : y1 + h, x1 : x1 + w] = res
		except Exception as e:
			print(e)
			print(x1, y1, x2, y2)

		return image, (x1, y1, x2, y2)


	def bus_call(self, bus, message):
		t = message.type
		if t == Gst.MessageType.EOS:
			sys.stdout.write("End-of-stream\n")
			#loop.quit()
		elif t==Gst.MessageType.WARNING:
			err, debug = message.parse_warning()
			sys.stderr.write("Warning: %s: %s\n" % (err, debug))
		elif t == Gst.MessageType.ERROR:
			err, debug = message.parse_error()
			sys.stderr.write("Error: %s: %s\n" % (err, debug))
			#loop.quit()
		return True

	def cb_newpad(self, decodebin, decoder_src_pad, data):
		print("In cb_newpad\n")
		caps = decoder_src_pad.get_current_caps()
		gststruct = caps.get_structure(0)
		gstname = gststruct.get_name()
		source_bin = data
		features = caps.get_features(0)

		# Need to check if the pad created by the decodebin is for video and not
		# audio.
		if gstname.find("video") != -1:
			# Link the decodebin pad only if decodebin has picked nvidia
			# decoder plugin nvdec_*. We do this by checking if the pad caps contain
			# NVMM memory features.
			print("features=", features)
			if features.contains("memory:NVMM"):
				# Get the source bin ghost pad
				bin_ghost_pad = source_bin.get_static_pad("src")
				if not bin_ghost_pad.set_target(decoder_src_pad):
					sys.stderr.write(
						"Failed to link decoder src pad to source bin ghost pad\n"
					)
			else:
				sys.stderr.write(
					" Error: Decodebin did not pick nvidia decoder plugin.\n")


	def decodebin_child_added(self, child_proxy, Object, name, user_data):
		print("Decodebin child added:", name, "\n")
		if name.find("decodebin") != -1:
			Object.connect("child-added", self.decodebin_child_added, user_data)


	def create_source_bin(self, index, uri):
		print("Creating source bin")

		# Create a source GstBin to abstract this bin's content from the rest of the
		# pipeline
		bin_name = "source-bin-%02d" % index
		print(bin_name)
		nbin = Gst.Bin.new(bin_name)
		if not nbin:
			sys.stderr.write(" Unable to create source bin \n")

		# Source element for reading from the uri.
		# We will use decodebin and let it figure out the container format of the
		# stream and the codec and plug the appropriate demux and decode plugins.
		uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
		if not uri_decode_bin:
			sys.stderr.write(" Unable to create uri decode bin \n")
		# We set the input uri to the source element
		uri_decode_bin.set_property("uri", uri)

		# Connect to the "pad-added" signal of the decodebin which generates a
		# callback once a new pad for raw data has beed created by the decodebin
		uri_decode_bin.connect("pad-added", self.cb_newpad, nbin)
		uri_decode_bin.connect("child-added", self.decodebin_child_added, nbin)

		# We need to create a ghost pad for the source bin which will act as a proxy
		# for the video decoder src pad. The ghost pad will not have a target right
		# now. Once the decode bin creates the video decoder and generates the
		# cb_newpad callback, we will set the ghost pad target to the video decoder
		# src pad.
		Gst.Bin.add(nbin, uri_decode_bin)
		bin_pad = nbin.add_pad(
			Gst.GhostPad.new_no_target(
				"src", Gst.PadDirection.SRC))
		if not bin_pad:
			sys.stderr.write(" Failed to add ghost pad in source bin \n")
			return None
		return nbin

	def play(self):
		self.pipeline.set_state(Gst.State.PLAYING)

	def stop(self):
		self.pipeline.set_state(Gst.State.NULL)