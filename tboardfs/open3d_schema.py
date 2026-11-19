"""Open3D v0.19.0 plugin_data.proto geometry descriptor, in a private pool.

Open3D: www.open3d.org
Copyright (c) 2018-2024 www.open3d.org

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from base64 import b64decode

from google.protobuf.descriptor_pool import DescriptorPool
from google.protobuf.message_factory import GetMessageClass

_GEOMETRY_SCHEMA = (
    "ChFwbHVnaW5fZGF0YS5wcm90bxISdGVuc29yYm9hcmQub3BlbjNkIrINChBPcGVuM0RQbHVn"
    "aW5EYXRhEg8KB3ZlcnNpb24YASABKAUSUwoTcHJvcGVydHlfcmVmZXJlbmNlcxgCIAMoCzI2"
    "LnRlbnNvcmJvYXJkLm9wZW4zZC5PcGVuM0RQbHVnaW5EYXRhLlByb3BlcnR5UmVmZXJlbmNl"
    "EkQKC2JhdGNoX2luZGV4GAMgASgLMi8udGVuc29yYm9hcmQub3BlbjNkLk9wZW4zRFBsdWdp"
    "bkRhdGEuQmF0Y2hJbmRleBp3ChFQcm9wZXJ0eVJlZmVyZW5jZRJQChFnZW9tZXRyeV9wcm9w"
    "ZXJ0eRgBIAEoDjI1LnRlbnNvcmJvYXJkLm9wZW4zZC5PcGVuM0RQbHVnaW5EYXRhLkdlb21l"
    "dHJ5UHJvcGVydHkSEAoIc3RlcF9yZWYYAiABKAQafwoJU3RhcnRTaXplEg0KBXN0YXJ0GAEg"
    "ASgEEgwKBHNpemUYAiABKAQSFQoNbWFza2VkX2NyYzMyYxgDIAEoDRIRCglhdXhfc3RhcnQY"
    "BCABKAQSEAoIYXV4X3NpemUYBSABKAQSGQoRYXV4X21hc2tlZF9jcmMzMmMYBiABKA0aYgoK"
    "QmF0Y2hJbmRleBIQCghmaWxlbmFtZRgBIAEoCRJCCgpzdGFydF9zaXplGAIgAygLMi4udGVu"
    "c29yYm9hcmQub3BlbjNkLk9wZW4zRFBsdWdpbkRhdGEuU3RhcnRTaXplIpMJChBHZW9tZXRy"
    "eVByb3BlcnR5EhQKEHZlcnRleF9wb3NpdGlvbnMQABISCg52ZXJ0ZXhfbm9ybWFscxABEhEK"
    "DXZlcnRleF9jb2xvcnMQAhIWChJ2ZXJ0ZXhfdGV4dHVyZV91dnMQAxIUChB0cmlhbmdsZV9p"
    "bmRpY2VzEAQSEwoPdHJpYW5nbGVfY29sb3JzEAUSFAoQdHJpYW5nbGVfbm9ybWFscxAGEhgK"
    "FHRyaWFuZ2xlX3RleHR1cmVfdXZzEAcSEAoMbGluZV9pbmRpY2VzEAgSDwoLbGluZV9jb2xv"
    "cnMQCRIcChhtYXRlcmlhbF9zY2FsYXJfbWV0YWxsaWMQQBIdChltYXRlcmlhbF9zY2FsYXJf"
    "cm91Z2huZXNzEEESHwobbWF0ZXJpYWxfc2NhbGFyX3JlZmxlY3RhbmNlEEISHgoabWF0ZXJp"
    "YWxfc2NhbGFyX2NsZWFyX2NvYXQQRBIoCiRtYXRlcmlhbF9zY2FsYXJfY2xlYXJfY29hdF9y"
    "b3VnaG5lc3MQRRIeChptYXRlcmlhbF9zY2FsYXJfYW5pc290cm9weRBGEiUKIW1hdGVyaWFs"
    "X3NjYWxhcl9hbWJpZW50X29jY2x1c2lvbhBHEiAKHG1hdGVyaWFsX3NjYWxhcl90cmFuc21p"
    "c3Npb24QSRIdChltYXRlcmlhbF9zY2FsYXJfdGhpY2tuZXNzEEsSJwojbWF0ZXJpYWxfc2Nh"
    "bGFyX2Fic29ycHRpb25fZGlzdGFuY2UQTBIeChptYXRlcmlhbF92ZWN0b3JfYmFzZV9jb2xv"
    "chBQEhoKFm1hdGVyaWFsX3ZlY3Rvcl9ub3JtYWwQUxIkCiBtYXRlcmlhbF92ZWN0b3JfYWJz"
    "b3JwdGlvbl9jb2xvchBYEiEKHW1hdGVyaWFsX3RleHR1cmVfbWFwX21ldGFsbGljEGASIgoe"
    "bWF0ZXJpYWxfdGV4dHVyZV9tYXBfcm91Z2huZXNzEGESJAogbWF0ZXJpYWxfdGV4dHVyZV9t"
    "YXBfcmVmbGVjdGFuY2UQYhIjCh9tYXRlcmlhbF90ZXh0dXJlX21hcF9jbGVhcl9jb2F0EGQS"
    "LQopbWF0ZXJpYWxfdGV4dHVyZV9tYXBfY2xlYXJfY29hdF9yb3VnaG5lc3MQZRIjCh9tYXRl"
    "cmlhbF90ZXh0dXJlX21hcF9hbmlzb3Ryb3B5EGYSKgombWF0ZXJpYWxfdGV4dHVyZV9tYXBf"
    "YW1iaWVudF9vY2NsdXNpb24QZxIlCiFtYXRlcmlhbF90ZXh0dXJlX21hcF90cmFuc21pc3Np"
    "b24QaRIiCh5tYXRlcmlhbF90ZXh0dXJlX21hcF90aGlja25lc3MQaxIfChttYXRlcmlhbF90"
    "ZXh0dXJlX21hcF9hbGJlZG8QbBIfChttYXRlcmlhbF90ZXh0dXJlX21hcF9ub3JtYWwQbxIp"
    "CiVtYXRlcmlhbF90ZXh0dXJlX21hcF9hYnNvcnB0aW9uX2NvbG9yEHQSJwojbWF0ZXJpYWxf"
    "dGV4dHVyZV9tYXBfYW9fcm91Z2hfbWV0YWwQdWIGcHJvdG8z"
)

_POOL = DescriptorPool()
_FILE = _POOL.AddSerializedFile(b64decode(_GEOMETRY_SCHEMA))
_DESCRIPTOR = _FILE.message_types_by_name["Open3DPluginData"]
GeometryIndex = GetMessageClass(_DESCRIPTOR)
GEOMETRY_PROPERTIES = _DESCRIPTOR.enum_types_by_name[
    "GeometryProperty"
].values_by_number
