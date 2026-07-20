import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DINOHead(nn.Module):
    """
    Projection head used during DINO self-supervised pretraining.

    The backbone produces image features.
    The DINO head maps those features to the DINO output space.
    """

    def __init__(
        self,
        in_dim,
        out_dim=65536,
        hidden_dim=2048,
        bottleneck_dim=256,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(
                bottleneck_dim,
                out_dim,
                bias=False,
            )
        )

        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)

        return x


class DINONetwork(nn.Module):
    """
    ViT backbone + DINO projection head.
    """

    def __init__(
        self,
        model_name="vit_tiny_patch16_224",
        out_dim=65536,
        hidden_dim=2048,
        bottleneck_dim=256,
        pretrained_backbone=False,
    ):
        super().__init__()

        # Default False preserves the original from-scratch pretraining
        # behaviour exactly. Set True (e.g. for a DINOv2 checkpoint tag
        # like "vit_small_patch14_dinov2.lvd142m") to start DINO
        # pretraining from an already-pretrained backbone instead of
        # random initialization — for continuing self-supervised
        # training on a new domain rather than learning from scratch.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained_backbone,
            num_classes=0,
            dynamic_img_size=True,
        )

        embed_dim = self.backbone.num_features

        self.head = DINOHead(
            in_dim=embed_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)

        return output


def build_dino_student_teacher(
    model_name="vit_tiny_patch16_224",
    out_dim=65536,
    hidden_dim=2048,
    bottleneck_dim=256,
    pretrained_backbone=False,
):
    """
    Create student and teacher DINO networks.

    The teacher starts with the same weights as the student.
    Teacher parameters do not receive gradients.

    pretrained_backbone:
        False (default): both start from a random backbone, exactly as
        before — every existing from-scratch DINO pretraining config is
        unaffected by this parameter's existence.
        True: both start from the named backbone's pretrained weights
        (e.g. a DINOv2 checkpoint) instead of random init, for continued
        pretraining on a new domain. The DINO projection head is always
        randomly initialized either way, since it's specific to this
        training setup and isn't part of any published backbone
        checkpoint.
    """

    student = DINONetwork(
        model_name=model_name,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        pretrained_backbone=pretrained_backbone,
    )

    teacher = DINONetwork(
        model_name=model_name,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        pretrained_backbone=pretrained_backbone,
    )

    teacher.load_state_dict(student.state_dict())

    for parameter in teacher.parameters():
        parameter.requires_grad = False

    return student, teacher