#Copyright 2022 Nathan Harwood
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import annotations
import numpy as np
from multiprocessing import Queue
from typing import NamedTuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .audiomodule import Buffer, AudioModule

CHANNELS_MONO = ['Left']
CHANNELS_STEREO = ['Left','Right']
CHANNELS_31 = ['Left','Right','LFE']
CHANNELS_QUAD = ['Left','Right','Left Surround','Right Surround']
CHANNELS_WAV_51 =  ['Left','Right','Center','LFE','Left Surround','Right Surround']
CHANNELS_DTS_51 =  ['Left','Right','Left Surround','Right Surround','Center','LFE']
CHANNELS_FILM_51 = ['Left','Center','Right','Left Surround','Right Surround','LFE']

LOW_QUALITY_RATE = 22050
CD_RATE = 44100
AV_RATE = 48000
DVD_RATE = 96000

PROBE_DATA = "probe_data"
""" The buffer has received data. """

MODULE_ERROR = "module_error"
""" The module is reporting an error condition has arisen. """

MODULE_LOG = "module_log"
""" The module wants to log something. """

AM_ERROR = -1
"""Indicates the module is in an erroneous state."""

AM_CONTINUE = 0
"""Indicates the module can continue to produce data."""

AM_COMPLETED = 1
"""Indicates the module has no more data to produce."""

AM_INPUT_REQUIRED = 2
"""Indicates the module requires input data to produce data."""

AM_CYCLIC_UNDERRUN = 3
"""Indicates the module requires input from a module earlier in the sequence.

Note that earlier means with a sequence number not greater than itself.
"""

def sw_dtype(self,sample_width:int=2):
    if sample_width == 1:
        return np.int8
    if sample_width == 2:
        return np.int16
    if sample_width == 4:
        return np.float32

def nice_frequency_str2(freq: float = 1.0):
    if freq > 1000:
        return f"{freq/1000.0:.0f}kHz"
    else:
        return f"{freq:.1f}Hz"

class ModId(NamedTuple):
    name:str
    id:int

class BufferId(NamedTuple):
    mod_id:ModId
    inout:str
    idx:int

class BufferData(NamedTuple):
    buffer: np.ndarray
    """Chunk of data."""

    remaining: int
    """Remaining data in the buffer."""


class Buffer:
    def __init__(self, channels: int = 1,
                 sample_rate: float = AV_RATE,
                 dtype=np.float32):
        self.channels = channels
        self.sample_rate = sample_rate
        self.dtype = dtype
        self.probe: BufferId = None
        self.observer: Queue = None
        self._incoming_chunks = []
        self._probe_chunks = []
        self._time:int = 0 # number of samples
        self._delayed_time:int = 0 # number of samples
        self._size = 0
        self.reset()

    def reset(self):
        """Empty the buffer and reset the buffer time.

        An existing probe is not canceled by this method."""

        self._buffer = self.get_empty_buffer()
        self._time = 0
        self._delayed_time = 0
    
    def _compact(self):
        if len(self._incoming_chunks)>0:
            self._buffer = np.concatenate([self._buffer]+self._incoming_chunks)
            self._incoming_chunks=[]

    def buffer(self):
        """ Provides direct access to the internal buffer. """

        self._compact()
        return self._buffer

    def append(self, buf: np.ndarray):
        """Append `buf` to the buffer.

        The `buf` is sent to the observer if a probe has been set.
        """

        if buf.shape[1] < self.channels:
            nbuf = self.get_zero_buffer(buf.shape[0])
            nbuf[:,0:buf.shape[1]]=buf
        else:
            nbuf = buf[:,0:self.channels]
        self._incoming_chunks.append(nbuf)
        self._size += len(nbuf)
        if self.probe:
            self._probe_chunks.append(nbuf)
            if len(self._probe_chunks)>4:
                self.observer.put((PROBE_DATA, (self.probe, 
                    np.concatenate(self._probe_chunks)), None))
                self._probe_chunks=[]

    def get_chunk(self, chunk_size: int = 1024) -> BufferData:
        """Returns up to `chunk_size` of data.

        If the buffer has not yet started advancing in time, returns an
        empty chunk of data of size `chunk_size`.
        """

        self._compact()
        actual_chunk_size = min([chunk_size, len(self._buffer)])
        x = np.split(self._buffer, [actual_chunk_size])
        self._buffer = x[1]
        self._time += len(x[0])
        self._size -= len(x[0])
        if self._time == 0:  # no data has arrived yet
            chunk = np.concatenate([x[0], 
                self.get_empty_buffer(chunk_size-actual_chunk_size)])
            self._delayed_time+= len(chunk)
        else:
            chunk = x[0]
        return BufferData(chunk, len(self._buffer))

    def get_all(self) -> np.ndarray:
        """Returns all the data from the buffer.
        
        Call only if data is known to be in the buffer.
        """

        self._compact()
        x = self._buffer
        self._time += len(x)
        self._buffer = self.get_empty_buffer()
        self._size = 0
        return x

    def set_probe(self, observer: Queue, buffer_id: BufferId = None):
        """Tells the buffer to report incomming data to an observer.

        The observer is running in another process and messages are sent
        via `multiprocessing.Queue`. Each observation is a tuple
        `('probe_data',(buffer_id,data))`.
        Call with `buffer_id=None` to cancel the probe."""

        self.probe = buffer_id
        self.observer = observer
        if self.probe:
            self._probe_chunks=[]

    def size(self):
        """Returns the size (number of samples) in the buffer.
        
        This is the sum of the `_buffer` len and the length of uncompacted
        incoming chunks."""

        return self._size
        #return len(self._buffer)+sum([len(x) for x in self._incoming_chunks])

    def get_empty_buffer(self, size: int = 0):
        """Returns an empty array of length `size` with the same shape as the buffer."""

        return np.zeros(shape=(size, self.channels), dtype=self.dtype)
        
    def get_zero_buffer(self, size: int = 0):
        """Returns a zero array of length `size` with the same shape as the buffer."""

        return np.zeros(shape=(size, self.channels), dtype=self.dtype)
        
    def get_time(self):
        """ Return the time duration of input signal consumed. """

        return self._time/self.sample_rate

    def get_delayed_time(self):
        """ Return the total delay incurred at the input. """

        return self._delayed_time/self.sample_rate




class AudioModule:
    """Base class for all audio modules."""

    name = "audio-module"
    """Name of the module shown to the user."""

    category = "none"
    """Category of the module shown to the user."""

    description = "Base class for all audio modules."
    """Description of the module shown to the user."""

    def __init__(self,
                 in_chs: list[int] = [1],
                 out_chs: list[int] = [1],
                 num_inputs: int = 1,
                 num_outputs: int = 1,
                 chunk_size: int = 1024,
                 sample_rate: float = AV_RATE,
                 dtype=np.float32,
                 observer: Queue = None,
                 mod_id: ModId = None,
                 polled: bool = False):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.sample_duration = 1.0/sample_rate
        self.dtype = dtype
        self.observer = observer
        self.mod_id = mod_id
        self.polled = polled
        self.out_bufs: list[Buffer] = []
        self.in_bufs: list[Buffer] = []
        self.out_modules: list[list[tuple[AudioModule, int]]] = [[]]
        self.in_modules: list[tuple[__class__,int]] = []
        self.sequence=-1
        self.configure_io(num_inputs, num_outputs, in_chs, out_chs)

    def configure_io(self, num_inputs: int, num_outputs: int,
                     in_chs: list[int], out_chs: list[int]):
        """Configure the module's inputs and outputs.

        The module *must* be disconnected from all other modules before
        calling this method.
        """

        self.out_chs = out_chs
        self.in_chs = in_chs
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        if self.num_outputs > 0:
            self.out_modules: list[list[tuple[AudioModule, int]]] = [
                [] for _ in range(self.num_outputs)]
        else:
            self.out_modules = None
        if self.num_inputs > 0:
            self.in_modules: list[tuple[AudioModule,int]] = [None]*self.num_inputs
        else:
            self.in_modules = None
        self.out_bufs = []
        self.in_bufs = []
        for i in range(self.num_outputs):
            self.out_bufs.append(Buffer(self.out_chs[i],
                                        sample_rate=self.sample_rate,
                                        dtype=self.dtype))
        for i in range(self.num_inputs):
            self.in_bufs.append(Buffer(self.in_chs[i],
                                       sample_rate=self.sample_rate,
                                       dtype=self.dtype))

    def __str__(self):
        if self.num_inputs > 0:
            input_sigs = [f":{x}c" for x in self.in_chs]
        else:
            input_sigs = ":none"
        if self.num_outputs > 0:
            output_sigs = [f":{x}c" for x in self.out_chs]
        else:
            output_sigs = ":none"
        return f"{self.name} inputs{''.join(input_sigs)} outputs{''.join(output_sigs)}"

    def receive_signal(self, signal: np.ndarray, in_idx: int = 0):
        """Append the signal to the module's input given by `in_idx`."""

        self.in_bufs[in_idx].append(signal)

    def get_input_chunk(self, in_idx: int = 0,
                        custom_chunk_size: int = None):
        """Retrieve a chunk of data from the input given by `in_idx`.

        The `custom_chunk_size` can be used to override the default `chunk_size`."""

        if custom_chunk_size:
            chunk_size = custom_chunk_size
        else:
            chunk_size = self.chunk_size
        return self.in_bufs[in_idx].get_chunk(chunk_size)

    def connect(self, module: AudioModule, module_in_idx: int = 0, out_idx: int = 0):
        """Connect this module's output `out_idx` to the given module's input `module_in_idx`."""

        if module_in_idx >= module.num_inputs:
            raise Exception(
                f"Signal input {module_in_idx} does not exist on module {module}")
        if module.in_modules[module_in_idx] != None:
            raise Exception(
                f"Input {module_in_idx} already has a module attached to it {module}")
        self.out_modules[out_idx].append((module, module_in_idx))
        module.in_modules[module_in_idx] = (self, out_idx)

    def disconnect(self, out_idx: int, to_mod: AudioModule, in_idx: int):
        """Disconnect the output `out_idx` going to the given module input `in_idx`."""

        try:
            self.out_modules[out_idx].remove((to_mod, in_idx))
        except:
            pass  # ignore ValueError when connection does not exist.
        to_mod.in_modules[in_idx] = None

    def flush_all_outputs(self):
        """Flushes all output buffers, sending data to downstream modules."""

        for out_idx in range(self.num_outputs):
            final_chunk = self.out_bufs[out_idx].get_all()
            for module, in_idx in self.out_modules[out_idx]:
                module.receive_signal(final_chunk, in_idx)

    def output_chunks(self, out_idx: int = 0):
        """Send the largest multiple of `chunk_size` data from output `out_idx`."""

        remaining = self.out_bufs[out_idx].size()
        while remaining >= self.chunk_size:
            multi_chunk = int(remaining / self.chunk_size)*self.chunk_size
            output_chunk, remaining = self.out_bufs[out_idx].get_chunk(
                multi_chunk)
            for module, in_idx in self.out_modules[out_idx]:
                module.receive_signal(output_chunk, in_idx)

    def input_pending(self):
        """Returns True if the module should proceed to produce a chunk of data.
        
        This is true if at least one input has at least one chunk of data available,
        or if all inputs are not at a higher sequence.
        """

        higher_sequence=False
        for idx in range(self.get_num_inputs()):
            buf = self.get_in_buf(idx)
            if buf.size() >= self.chunk_size:
                return True
            x = self.get_in_modules(idx)
            if x:
                mod,out_idx=x
                if mod.sequence>=0:
                    higher_sequence |= (mod.get_sequence()>self.sequence)
                else:
                    higher_sequence = True
            else:
                higher_sequence = True
        return not higher_sequence

    def input_underrun(self):
        """Returns `(a,b)` where `a` is true if at least one input is at time > 0.0 *without*
        at least one chunk of data available.
        
        If `a` is true then `b` is true if the input comes from an earlier sequence module
        which indicates that the module should return `AM_CYCLIC_UNDERRUN`,
        otherwise `b` is false.
        """

        for idx in range(self.get_num_inputs()):
            buf = self.get_in_buf(idx)
            if buf.size() < self.chunk_size and buf.get_time() > 0:
                x = self.get_in_modules(idx)
                if x:
                    mod,out_idx = x
                    return True, (self.sequence >= 0 and mod.get_sequence()<=self.sequence)
                else:
                    return True, False
        return False, False

    def send_signal(self, signal: np.ndarray, out_idx: int = 0):
        """Send the signal to output `out_idx`.

        This writes the signal to the output buffer and if the
        module is not polled it sends as many chunks of data as
        possible to modules connected to output `out_idx`.
        """

        self.out_bufs[out_idx].append(signal)
        if not self.polled:
            self.output_chunks(out_idx)

    def get_widget_params(self):
        """Return widget parameter information.

        Widget parameters can be set by the user."""

        return {
            'meta_order': [],
            'name': {
                'name': 'Name',
                'value': self.name
            }
        }

    def set_sequence(self,sequence:int=0):
        self.sequence=sequence

    def get_sequence(self):
        return self.sequence

    def set_in_buf_probe(self, observer: Queue, buffer_id: BufferId):
        """Set a probe on the input buffer."""

        self.in_bufs[buffer_id.idx].set_probe(observer, buffer_id)

    def unset_in_buf_probe(self, observer: Queue, buffer_id: BufferId):
        """Remove a buffer probe from the input buffer."""

        self.in_bufs[buffer_id.idx].set_probe(observer, None)

    def set_out_buf_probe(self, observer: Queue, buffer_id: BufferId):
        """Set a probe on the output buffer."""

        self.out_bufs[buffer_id.idx].set_probe(observer, buffer_id)

    def unset_out_buf_probe(self, observer: Queue, buffer_id: BufferId):
        """Remove a probe on the output buffer."""

        self.out_bufs[buffer_id.idx].set_probe(observer, None)

    def get_empty_out_buf(self, out_idx: int = 0):
        """Get an empty array with the shape of the output buffer."""

        return self.out_bufs[out_idx].get_empty_buffer()

    def get_empty_in_buf(self, in_idx: int = 0):
        """Get an empty array with the shape of the input buffer."""

        return self.in_bufs[in_idx].get_empty_buffer()

    def get_num_inputs(self) -> int:
        """Get the number of inputs."""

        return self.num_inputs

    def get_num_outputs(self) -> int:
        """Get the number of outputs."""

        return self.num_outputs

    def get_out_modules(self, out_idx: int = 0) -> list[tuple[AudioModule, int]]:
        """Get the list of connected modules at the given output."""

        return self.out_modules[out_idx]

    def get_in_modules(self, in_idx: int = 0) -> tuple[AudioModule,int]:
        """Get the module connected to the given input."""

        return self.in_modules[in_idx]

    def get_in_chs(self, in_idx: int = 0) -> int:
        """Get the number of channels of the given input."""

        return self.in_chs[in_idx]

    def get_out_chs(self, out_idx: int = 0) -> int:
        """Get the number of channels of the given output."""

        return self.out_chs[out_idx]

    def get_name(self) -> str:
        """Get the name of the module."""

        return self.name

    def get_in_buf(self, in_idx: int = 0) -> Buffer:
        """Get the given input buffer."""

        return self.in_bufs[in_idx]

    def get_out_buf(self, out_idx: int = 0) -> Buffer:
        """Get the given output buffer."""

        return self.out_bufs[out_idx]

    def set_widget_params(self, params):
        """Set the module's parameters."""
        self.name = params['name']['value']

    def start(self):
        """Audio engine has started.

        Input data will be provided as required.
        Make sure to call this `super()` method if overriding.
        """

        pass

    def stop(self):
        """Audio engine is stopping (pausing).

        More input data will not become available until the module is
        started again. Make sure to call this `super()` method if
        overriding.
        """

        pass

    def open(self):
        """Prepare the module (prior to starting).

        Input and output buffers are cleared. The module
        should reset all internal state to intitial conditions.
        Make sure to call this `super()` method if overriding.
        """

        for buffer in self.in_bufs:
            buffer.reset()
        for buffer in self.out_bufs:
            buffer.reset()

    def close(self):
        """Close/cleanup the module (after stopping).

        Make sure to call this `super()` method if overriding.
        """

        pass

    def reset(self):
        """Set source modules to start again.
        
        Make sure to call this `super()` method if overriding."""

        pass

    async def next_chunk(self) -> AM_COMPLETED | AM_CONTINUE | AM_INPUT_REQUIRED | AM_ERROR | AM_CYCLIC_UNDERRUN:
        """Process/produce a chunk of data.

        For modules that produce data, the module *must* produce exactl
        one chunk (of size `chunk_size`)
        of data on all outputs or otherwise return `AM_INPUT_REQUIRED` if this is
        not possible. The module *may* produce less than one chunk of data, but must
        in this case still return `AM_INPUT_REQUIRED`, and continue in this way until
        exactly one chunk of data has been produced, in which case `AM_CONTINUE` is
        returned. If `AM_INPUT_REQUIRED`
        is not returned then either `AM_COMPLETED` (no more data to produce) or
        `AM_CONTINUE` (produced a chunk of data) must be returned. The
        module can return `AM_ERROR` to indicate that the module is
        unable to correctly produce data.
        For modules that do not produce data, the module should return
        `AM_INPUT_REQUIRED` if more data is _urgently_ required on the input (to
        meet real-time playback requirements) or `AM_CONTINUE` if additional
        input data is not urgently required.
        """

        return AM_COMPLETED

    def process_all(self):
        """Process all available input data and produce output data."""

        pass

    def get_rate_change(self)->float:
        """For internal modules only, return ratio of output samples produced to input samples consumed.
        
        Producing twice as many output samples as input samples is a rate of 2.
        """

        return 1.0

    def get_status(self):
        """Return status information to display."""

        return {
            'topleft':"",
            'top':"",
            'topright':"",
            'bottomleft':"",
            'bottom':"", 
            'bottomright':"" 
        }


AudioModules: dict[str, AudioModule] = dict()
"""Audio modules that are registered for use.

Mapping from the module's class `__name__` to the module's `__class__`."""


def audiomod(mod: AudioModule.__class__) -> AudioModule.__class__:
    """Register the module for use."""

    AudioModules[mod.__name__] = mod
    return mod
