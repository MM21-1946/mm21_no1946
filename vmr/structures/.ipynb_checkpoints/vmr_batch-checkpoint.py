from dataclasses import dataclass
import torch

# temporal localization grounding 
@dataclass
class VMRBatch(object):
    # frames: list # [ImageList]
    feats: torch.tensor
    segmasks: torch.tensor
    queries: torch.tensor
    adjmats: torch.tensor
    constimasks: torch.tensor
    wordmasks: torch.tensor
    def to(self, device):
        # self.frames = [f.to(device) for f in self.frames]
        self.feats = self.feats.to(device)
        self.segmasks = self.segmasks.to(device)
        self.queries = self.queries.to(device)
        self.adjmats = self.adjmats.to(device) if self.adjmats is not None else None
        self.constimasks = self.constimasks.to(device) if self.constimasks is not None else None
        self.wordmasks = self.wordmasks.to(device)
        return self

@dataclass
class VMRGroundTruth(object):
    # frames: list # [ImageList]
    # torch.stack(ious2d), torch.stack(s_pos), torch.stack(e_pos)
    ious2d: torch.tensor
    s_pos_normed: torch.tensor
    e_pos_normed: torch.tensor
    targetmask: torch.tensor
    ious1d: torch.tensor
    ious1dmask: torch.tensor

    def to(self, device):
        # self.frames = [f.to(device) for f in self.frames]
        self.ious2d = self.ious2d.to(device) if self.ious2d is not None else None
        self.s_pos_normed = self.s_pos_normed.to(device) if self.s_pos_normed is not None else None
        self.e_pos_normed = self.e_pos_normed.to(device) if self.e_pos_normed is not None else None
        self.targetmask = self.targetmask.to(device) if self.targetmask is not None else None
        self.ious1d = self.ious1d.to(device) if self.ious1d is not None else None
        self.ious1dmask = self.ious1dmask.to(device) if self.ious1dmask is not None else None
        return self