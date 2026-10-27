"""Auditory input pathway: video soundtrack -> Johnston's-organ model.

`hearing.audio_source` decodes the audio track of the input video;
`hearing.johnstons_organ` turns it into simplified auditory-receptor drive.
Everything here is labeled BIOLOGY / SIMULATION / ASSUMPTION in the module
docstrings, matching the rest of the project (see docs/SIMULATION_NOTES.md).
"""

from hearing.audio_source import AudioTrack, has_audio_stream
from hearing.johnstons_organ import AuditoryReceptorModel

__all__ = ["AudioTrack", "has_audio_stream", "AuditoryReceptorModel"]
