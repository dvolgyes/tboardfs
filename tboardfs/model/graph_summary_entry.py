from dataclasses import dataclass

from tboardfs.model.binary_entry import BinaryEntry


@dataclass
class GraphSummaryEntry(BinaryEntry):
    """Index entry for a TF2 graph summary: a Keras config or a RunMetadata.

    :ivar converts: whether the summary converts to a GraphDef file
    :ivar profiles: whether the summary has a profile text file
    """

    converts: bool = False
    profiles: bool = False
