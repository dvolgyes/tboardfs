from dataclasses import dataclass

from tboardfs.model.graph_entry import GraphEntry


@dataclass
class MetaGraphEntry(GraphEntry):
    """Index entry for a TF1 MetaGraphDef payload.

    :ivar readable: whether the MetaGraphDef parses, so that its graph reads
    """

    readable: bool = False
