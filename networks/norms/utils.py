def parse_normalization(norm_name,
                        affine,
                        num_groups=None,
                        num_styles=None):
    if norm_name == "instance_cond":
        return "instance_cond", {"num_styles": num_styles, 'affine': affine}
    elif norm_name == "instance":
        return "instance", {"affine": affine}
    elif norm_name == "layer":
        return "layer", {"elementwise_affine": affine}
    elif norm_name == "batch":
        return "batch", {"affine": affine}
    elif norm_name == "group":
        return "group", {"affine": affine, "num_groups": num_groups}
    else:
        raise ValueError("Normalization {} not implemented. Please chose another model.".format(norm_name))
