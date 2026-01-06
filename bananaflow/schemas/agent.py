from typing import List, Dict, Any, Union, Literal
from pydantic import BaseModel, Field
from typing import Optional

class AddNodeOp(BaseModel):
    op: Literal["add_node"]
    node: Dict[str, Any]

class AddConnOp(BaseModel):
    op: Literal["add_connection"]
    connection: Dict[str, Any]

class UpdateNodeOp(BaseModel):
    op: Literal["update_node"]
    id: str
    data: Dict[str, Any] = Field(default_factory=dict)

class DeleteNodeOp(BaseModel):
    op: Literal["delete_node"]
    id: str

class DeleteConnOp(BaseModel):
    op: Literal["delete_connection"]
    id: str

class SelectNodesOp(BaseModel):
    op: Literal["select_nodes"]
    ids: List[str] = Field(default_factory=list)

class SetViewportOp(BaseModel):
    op: Literal["set_viewport"]
    viewport: Dict[str, float] = Field(default_factory=dict)

PatchOp = Union[
    AddNodeOp, AddConnOp, UpdateNodeOp,
    DeleteNodeOp, DeleteConnOp, SelectNodesOp, SetViewportOp
]

class AgentOut(BaseModel):
    patch: List[PatchOp]
    summary: str = ""
    thought: str = ""

class SelectedArtifact(BaseModel):
    url: str
    kind: Optional[str] = "image"
    fromNodeId: Optional[str] = None
    createdAt: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None

class AgentRequest(BaseModel):
    prompt: str = Field(default="")
    current_nodes: List[Dict[str, Any]] = Field(default_factory=list)
    current_connections: List[Dict[str, Any]] = Field(default_factory=list)
    selected_artifact: Optional[Dict[str, Any]] = None

    # ✅ 多画布关键字段
    canvas_id: Optional[str] = None
    thread_id: Optional[str] = None