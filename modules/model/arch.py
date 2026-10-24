# model architecture

# =================================================================================
# DCCRN architecture
# =================================================================================
# model_nlayer_lstm
# 3.7M (3.671)
DCCRN_12_512 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        ((1,  16, (5, 2), (2, 1), (1, 1)),
         (16,  32, (5, 2), (2, 1), (2, 1)),
         (32,  64, (5, 2), (2, 1), (2, 1)),
         (64, 128, (5, 2), (2, 1), (2, 1)),
         (128, 128, (5, 2), (2, 1), (2, 1)),
         (128, 128, (5, 2), (2, 1), (2, 1))),
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        ((256, 128, (5, 2), (2, 1), (2, 0), (1, 0)),
         (256, 128, (5, 2), (2, 1), (2, 0), (1, 0)),
         (256,  64, (5, 2), (2, 1), (2, 0), (1, 0)),
         (128,  32, (5, 2), (2, 1), (2, 0), (1, 0)),
         (64,  16, (5, 2), (2, 1), (2, 0), (1, 0)),
         (32,   1, (5, 2), (2, 1), (1, 0), (0, 0), True))
}

DCCRN_12_512_m2 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        ((1,  16, (5, 2), (4, 1), (1, 1)),
         (16,  32, (5, 2), (2, 1), (2, 1)),
         (32,  64, (5, 2), (2, 1), (2, 1)),
         (64, 128, (5, 2), (2, 1), (2, 1)),
         (128, 128, (5, 2), (2, 1), (2, 1)),
         (128, 128, (5, 2), (2, 1), (2, 1))),
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        ((256, 128, (5, 2), (2, 1), (2, 0), (1, 0)),
         (256, 128, (5, 2), (2, 1), (2, 0), (1, 0)),
         (256,  64, (5, 2), (2, 1), (2, 0), (1, 0)),
         (128,  32, (5, 2), (2, 1), (2, 0), (1, 0)),
         (64,  16, (5, 2), (2, 1), (2, 0), (1, 0)),
         (32,   1, (5, 2), (4, 1), (0, 0), (0, 0), True))
}

DCCRN_12_512_m4 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        ((1,  16, (5, 2), (4, 1), (1, 1)),
         (16,  32, (5, 2), (4, 1), (2, 1)),
         (32,  64, (5, 2), (2, 1), (2, 1)),
         (64, 128, (5, 2), (2, 1), (2, 1)),
         (128, 128, (5, 2), (2, 1), (2, 1)),
         (128, 128, (5, 2), (2, 1), (2, 1))),
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        ((256, 128, (5, 2), (2, 1), (2, 0), (1, 0)),
         (256, 128, (5, 2), (2, 1), (2, 0), (1, 0)),
         (256,  64, (5, 2), (2, 1), (2, 0), (1, 0)),
         (128,  32, (5, 2), (2, 1), (2, 0), (1, 0)),
         (64,  16, (5, 2), (4, 1), (2, 0), (1, 0)),
         (32,   1, (5, 2), (4, 1), (0, 0), (0, 0), True))
}


# =================================================================================
# CRN architecture
# =================================================================================
# model_nlayer_lstm
CRN_10_128 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        ((1,  64, (5, 2), (3, 1)),
         (64, 128, (5, 2), (3, 1)),
         (128, 128, (5, 2), (3, 1)),
         (128, 128, (5, 2), (3, 1)),
         (128, 128, (5, 2), (3, 1))),
    # (input_size, hidden_size, num_layers)
    'lstm': (128, 128, 2),
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        ((256, 128, (5, 2), (3, 1)),
         (256, 128, (5, 2), (3, 1)),
         (256, 128, (5, 2), (3, 1)),
         (256,  64, (5, 2), (3, 1)),
         (128,   1, (5, 2), (3, 1), (0, 0), (0, 0), True))
}

CRN_10_256 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        ((1,  64, (5, 2), (3, 1)),
         (64, 128, (5, 2), (3, 1)),
         (128, 128, (5, 2), (3, 1)),
         (128, 128, (5, 2), (3, 1)),
         (128, 256, (5, 2), (3, 1))),
    # (input_size, hidden_size, num_layers)
    'lstm': (256, 256, 2),
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        ((512, 128, (5, 2), (3, 1)),
         (256, 128, (5, 2), (3, 1)),
         (256, 128, (5, 2), (3, 1)),
         (256,  64, (5, 2), (3, 1)),
         (128,   1, (5, 2), (3, 1), (0, 0), (0, 0), True))
}

CRN_10_512 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        ((1,  64, (5, 2), (3, 1)),
         (64, 128, (5, 2), (3, 1)),
         (128, 128, (5, 2), (3, 1)),
         (128, 128, (5, 2), (3, 1)),
         (128, 512, (5, 2), (3, 1))),
    # (input_size, hidden_size, num_layers)
    'lstm': (512, 512, 2),
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        ((1024, 128, (5, 2), (3, 1)),
         (256, 128, (5, 2), (3, 1)),
         (256, 128, (5, 2), (3, 1)),
         (256,  64, (5, 2), (3, 1)),
         (128,   1, (5, 2), (3, 1), (0, 0), (0, 0), True))
}

# 1.865M
CRN_12_128 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        ((1,  32, (5, 2), (2, 1)),
         (32,  64, (5, 2), (2, 1)),
         (64,  96, (5, 2), (2, 1)),
         (96, 128, (5, 2), (2, 1)),
         (128, 128, (5, 2), (2, 1)),
         (128, 128, (5, 2), (2, 1))),
    # (input_size, hidden_size, num_layers)
    'lstm': (128, 128, 2),
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        ((256, 128, (5, 2), (2, 1)),
         (256, 128, (5, 2), (2, 1)),
         (256,  96, (5, 2), (2, 1)),
         (192,  64, (5, 2), (2, 1)),
         (128,  32, (5, 2), (2, 1)),
         (64,   1, (5, 2), (2, 1), (0, 0), (0, 0), True))
}

CRN_12_128_FC = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        ((1,  32, (5, 2), (2, 1)),
         (32,  60, (5, 2), (2, 1)),
         (60,  96, (5, 2), (2, 1)),
         (96, 128, (5, 2), (2, 1)),
         (128, 128, (5, 2), (2, 1)),
         (128, 128, (5, 2), (2, 1))),
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        ((256, 128, (5, 2), (2, 1)),
         (256, 128, (5, 2), (2, 1)),
         (256,  96, (5, 2), (2, 1)),
         (192,  60, (5, 2), (2, 1)),
         (120,  32, (5, 2), (2, 1)),
         (64,   1, (5, 2), (2, 1), (0, 0), (0, 0), True))
}

CRN_6_128_DPRNN = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        ((1,  32, (5, 2), (2, 1)),
         (32,  64, (5, 2), (2, 1)),
         (64, 128, (5, 2), (2, 1))),
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        ((256,  64, (5, 2), (2, 1)),
         (128,  32, (5, 2), (2, 1)),
         (64,   1, (5, 2), (2, 1), (0, 0), (0, 0), True))
}

UNet_8_256 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        [[0,  32, (5, 2), (2, 1), (1, 1)],
         (32,  64, (5, 2), (2, 1), (2, 1)),
         (64, 128, (5, 2), (2, 1), (2, 1)),
         (128, 256, (5, 2), (2, 1), (2, 1))],
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        [(512, 128, (5, 2), (2, 1), (2, 0), (1, 0)),
         (256,  64, (5, 2), (2, 1), (2, 0), (1, 0)),
         (128,  32, (5, 2), (2, 1), (2, 0), (1, 0)),
         [64,   0, (5, 2), (2, 1), (1, 0), (0, 0), True]]
}

UNet_6_128 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        [[0,  32, (5, 2), (2, 1), (1, 1)],
         (32,  64, (5, 2), (2, 1), (2, 1)),
         (64, 128, (5, 2), (2, 1), (2, 1))],
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        [(256,  64, (5, 2), (2, 1), (2, 0), (1, 0)),
         (128,  32, (5, 2), (2, 1), (2, 0), (1, 0)),
         [64,   0, (5, 2), (2, 1), (1, 0), (0, 0), True]]
}

UNet_6_64 = {
    # (in_channels, out_channels, kernel_size, stride, padding, causal)
    'encoder':
        [[0,  16, (5, 2), (2, 1), (1, 1)],
         (16, 32, (5, 2), (2, 1), (2, 1)),
         (32, 64, (1, 1), (1, 1), (0, 0), False)],  # 1*1, causal=False
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last, causal)
    'decoder':
        [(128, 32, (1, 1), (1, 1), (0, 0), (0, 0), False, False),  # 1*1, causal=False
         (64,  16, (5, 2), (2, 1), (2, 0), (1, 0)),
         [32,   0, (5, 2), (2, 1), (1, 0), (0, 0), True]]
}

UNet_6_32 = {
    # (in_channels, out_channels, kernel_size, stride, padding, causal)
    'encoder':
        [[0,  8, (5, 2), (2, 1), (1, 1)],
         (8, 16, (5, 2), (2, 1), (2, 1)),
         (16, 32, (5, 2), (2, 1), (2, 1))],
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last, causal)
    'decoder':
        [(64, 16, (5, 2), (2, 1), (2, 0), (1, 0)),
         (32,  8, (5, 2), (2, 1), (2, 0), (1, 0)),
         [16,  0, (5, 2), (2, 1), (1, 0), (0, 0), True]]
}

# UNet_6_16 = {
#     # (in_channels, out_channels, kernel_size, stride, padding, causal)
#     'encoder':
#         [[ 0,  4, (5, 2), (2, 1), (1, 1)],
#          ( 4,  8, (5, 2), (2, 1), (2, 1)),
#          ( 8, 16, (5, 2), (2, 1), (2, 1))],
#     # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last, causal)
#     'decoder':
#         [(32,  8, (5, 2), (2, 1), (2, 0), (1, 0)),
#          (16,  4, (5, 2), (2, 1), (2, 0), (1, 0)),
#          [ 8,  0, (5, 2), (2, 1), (1, 0), (0, 0), True]]
# }

UNet_6_16 = {
    # (in_channels, out_channels, kernel_size, stride, padding, causal)
    'encoder':
        [[0,  4, (5, 2), (4, 1), (1, 1)],
         (4,  8, (5, 2), (2, 1), (2, 1)),
         (8, 16, (5, 2), (2, 1), (2, 1))],
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last, causal)
    'decoder':
        [(32,  8, (5, 2), (2, 1), (2, 0), (1, 0)),
         (16,  4, (5, 2), (2, 1), (2, 0), (1, 0)),
         [8,  0, (5, 2), (4, 1), (0, 0), (0, 0), True]]
}

UNet1_8_256 = UNet_8_256
UNet1_6_128 = UNet_6_128
UNet1_6_64 = UNet_6_64

TNet_6_64 = {
    # (in_channels, out_channels, kernel_size, stride, padding, causal)
    'encoder':
        [[0,  16, (5, 2), (2, 1), (1, 1)],
         (16,  32, (5, 2), (2, 1), (2, 1)),
         (32,  64, (1, 1), (1, 1), (0, 0), False)],  # 1*1, causal=False
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last, causal)
    'decoder':
        [(128,  32, (1, 1), (1, 1), (0, 0), (0, 0), False, False),  # 1*1, causal=False
         (64,  16, (5, 2), (2, 1), (2, 0), (1, 0)),
         [32,   0, (5, 2), (2, 1), (1, 0), (0, 0), True]]
}

# 1.6M
TNet_6_128 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        [[0,  32, (5, 2), (2, 1), (1, 1)],
         (32,  64, (5, 2), (2, 1), (2, 1)),
         (64, 128, (5, 2), (2, 1), (2, 1))],
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        [(256,  64, (5, 2), (2, 1), (2, 0), (1, 0)),
         (128,  32, (5, 2), (2, 1), (2, 0), (1, 0)),
         [64,   0, (5, 2), (2, 1), (1, 0), (0, 0), True]]
}

merge_net_8 = [
    # (in_channels, out_channels, kernel_size, stride, padding)
    (3, 8, (5, 2), (1, 1), (2, 1)),
    (8, 4, (5, 2), (1, 1), (2, 1)),
    (4, 1, (5, 2), (1, 1), (2, 1))
]

merge_net_6 = [
    # (in_channels, out_channels, kernel_size, stride, padding)
    (3, 6, (5, 2), (1, 1), (2, 1)),
    (6, 3, (5, 2), (1, 1), (2, 1)),
    (3, 1, (5, 2), (1, 1), (2, 1))
]

merge_net_3 = [  # set causal=False !
    # (in_channels, out_channels, kernel_size, stride, padding),
    (3, 3, (5, 3), (1, 1), (2, 1)),
    (3, 3, (5, 3), (1, 1), (2, 1)),
    (3, 1, (5, 3), (1, 1), (2, 1))
]

# 1.117M
TNet_6_128_2 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        [[0,  32, (5, 2), (2, 1), (1, 1)],
         (32,  64, (5, 2), (2, 1), (2, 1)),
         (64, 128, (1, 1), (1, 1), (0, 0), False)],  # 1*1, causal=False
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        [(256,  64, (1, 1), (1, 1), (0, 0), (0, 0), False, False),  # 1*1, causal=False
         (128,  32, (5, 2), (2, 1), (2, 0), (1, 0)),
         [64,   0, (5, 2), (2, 1), (1, 0), (0, 0), True]]
}


# 1.5M
CRN_12_120 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        ((1,  32, (5, 2), (2, 1)),
         (32,  64, (5, 2), (2, 1)),
         (64,  64, (5, 2), (2, 1)),
         (64, 120, (5, 2), (2, 1)),
         (120, 120, (5, 2), (2, 1)),
         (120, 120, (5, 2), (2, 1))),
    # (input_size, hidden_size, num_layers)
    'lstm': (120, 120, 2),
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        ((240, 120, (5, 2), (2, 1)),
         (240, 120, (5, 2), (2, 1)),
         (240,  64, (5, 2), (2, 1)),
         (128,  64, (5, 2), (2, 1)),
         (128,  32, (5, 2), (2, 1)),
         (64,   1, (5, 2), (2, 1), (0, 0), (0, 0), True))
}

# 1.016M
CRN_12_96 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        ((1, 24, (5, 2), (2, 1)),
         (24, 48, (5, 2), (2, 1)),
         (48, 64, (5, 2), (2, 1)),
         (64, 96, (5, 2), (2, 1)),
         (96, 96, (5, 2), (2, 1)),
         (96, 96, (5, 2), (2, 1))),
    # (input_size, hidden_size, num_layers)
    'lstm': (96, 96, 2),
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        ((192, 96, (5, 2), (2, 1)),
         (192, 96, (5, 2), (2, 1)),
         (192, 64, (5, 2), (2, 1)),
         (128, 48, (5, 2), (2, 1)),
         (96, 24, (5, 2), (2, 1)),
         (48,  1, (5, 2), (2, 1), (0, 0), (0, 0), True))
}

# 513.937K
CRN_12_64 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        ((1, 16, (5, 2), (2, 1)),
         (16, 32, (5, 2), (2, 1)),
         (32, 64, (5, 2), (2, 1)),
         (64, 64, (5, 2), (2, 1)),
         (64, 64, (5, 2), (2, 1)),
         (64, 64, (5, 2), (2, 1))),
    # (input_size, hidden_size, num_layers)
    'lstm': (64, 64, 2),
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        ((128, 64, (5, 2), (2, 1)),
         (128, 64, (5, 2), (2, 1)),
         (128, 64, (5, 2), (2, 1)),
         (128, 32, (5, 2), (2, 1)),
         (64, 16, (5, 2), (2, 1)),
         (32,  1, (5, 2), (2, 1), (0, 0), (0, 0), True))
}

# 3.7M
CRN_12_256 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        ((1,  60, (5, 2), (2, 1)),
         (60, 128, (5, 2), (2, 1)),
         (128, 128, (5, 2), (2, 1)),
         (128, 128, (5, 2), (2, 1)),
         (128, 128, (5, 2), (2, 1)),
         (128, 256, (5, 2), (2, 1))),
    # (input_size, hidden_size, num_layers)
    'lstm': (256, 256, 2),
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        ((512, 128, (5, 2), (2, 1)),
         (256, 128, (5, 2), (2, 1)),
         (256, 128, (5, 2), (2, 1)),
         (256, 128, (5, 2), (2, 1)),
         (256,  60, (5, 2), (2, 1)),
         (120,   1, (5, 2), (2, 1), (0, 0), (0, 0), True))
}

# 3.7M LSTM with Full-Connect
CRN_12_256_FC = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        ((1,  16, (5, 2), (2, 1)),
         (16,  32, (5, 2), (2, 1)),
         (32,  76, (5, 2), (2, 1)),
         (76, 128, (5, 2), (2, 1)),
         (128, 256, (5, 2), (2, 1)),
         (256, 256, (5, 2), (2, 1))),
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        ((512, 256, (5, 2), (2, 1)),
         (512, 128, (5, 2), (2, 1)),
         (256,  76, (5, 2), (2, 1)),
         (152,  32, (5, 2), (2, 1)),
         (64,  16, (5, 2), (2, 1)),
         (32,   1, (5, 2), (2, 1), (0, 0), (0, 0), True))
}

# 3.0M
CRN_12_192 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        ((1,  60, (5, 2), (2, 1)),
         (60, 128, (5, 2), (2, 1)),
         (128, 128, (5, 2), (2, 1)),
         (128, 128, (5, 2), (2, 1)),
         (128, 128, (5, 2), (2, 1)),
         (128, 190, (5, 2), (2, 1))),
    # (input_size, hidden_size, num_layers)
    'lstm': (190, 190, 2),
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        ((380, 128, (5, 2), (2, 1)),
         (256, 128, (5, 2), (2, 1)),
         (256, 128, (5, 2), (2, 1)),
         (256, 128, (5, 2), (2, 1)),
         (256,  60, (5, 2), (2, 1)),
         (120,   1, (5, 2), (2, 1), (0, 0), (0, 0), True))
}

# 2.0M
CRN_12_148 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        ((1,  32, (5, 2), (2, 1)),
         (32,  64, (5, 2), (2, 1)),
         (64,  96, (5, 2), (2, 1)),
         (96, 128, (5, 2), (2, 1)),
         (128, 128, (5, 2), (2, 1)),
         (128, 145, (5, 2), (2, 1))),
    # (input_size, hidden_size, num_layers)
    'lstm': (145, 145, 2),
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        ((290, 128, (5, 2), (2, 1)),
         (256, 128, (5, 2), (2, 1)),
         (256,  96, (5, 2), (2, 1)),
         (192,  64, (5, 2), (2, 1)),
         (128,  32, (5, 2), (2, 1)),
         (64,   1, (5, 2), (2, 1), (0, 0), (0, 0), True))
}

# model_nlayer_lstm_feature
CRN_10_256_mel80 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        ((1,  16, (3, 2), (2, 1), (1, 1)),
         (16,  32, (3, 2), (2, 1), (1, 1)),
         (32,  64, (3, 2), (2, 1), (1, 1)),
         (64, 128, (3, 2), (2, 1), (0, 1)),
         (128, 256, (3, 2), (2, 1), (0, 1))),
    # (input_size, hidden_size, num_layers)
    'lstm':
        (256, 256, 2),
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        ((512, 128, (3, 2), (2, 1), (0, 0), (1, 0), False),
         (256,  64, (3, 2), (2, 1), (0, 0), (1, 0), False),
         (128,  32, (3, 2), (2, 1), (1, 0), (1, 0), False),
         (64,  16, (3, 2), (2, 1), (1, 0), (1, 0), False),
         (32,   1, (3, 2), (2, 1), (1, 0), (1, 0), True))
}

CRN_10_256_mel80 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        ((1,  16, (3, 2), (2, 1), (1, 1)),
         (16,  32, (3, 2), (2, 1), (1, 1)),
         (32,  64, (3, 2), (2, 1), (1, 1)),
         (64, 128, (3, 2), (2, 1), (0, 1)),
         (128, 256, (3, 2), (2, 1), (0, 1))),
    # (input_size, hidden_size, num_layers)
    'lstm':
        (256, 256, 2),
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        ((512, 128, (3, 2), (2, 1), (0, 0), (1, 0), False),
         (256,  64, (3, 2), (2, 1), (0, 0), (1, 0), False),
         (128,  32, (3, 2), (2, 1), (1, 0), (1, 0), False),
         (64,  16, (3, 2), (2, 1), (1, 0), (1, 0), False),
         (32,   1, (3, 2), (2, 1), (1, 0), (1, 0), True))
}

CRN_8_128_mel80_DPRNN = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        ((1,  16, (3, 2), (2, 1), (1, 1)),
         (16,  32, (3, 2), (2, 1), (1, 1)),
         (32,  64, (3, 2), (2, 1), (1, 1)),
         (64, 128, (3, 2), (2, 1), (0, 1))),
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        ((256,  64, (3, 2), (2, 1), (0, 0), (1, 0), False),
         (128,  32, (3, 2), (2, 1), (1, 0), (1, 0), False),
         (64,  16, (3, 2), (2, 1), (1, 0), (1, 0), False),
         (32,   1, (3, 2), (2, 1), (1, 0), (1, 0), True))
}


# =================================================================================
# Discriminator architecture
# =================================================================================
# model_nlayer_nchannel_nfft
CNN_6_128_512 = [
    # (in_channels, out_channels, kernel_size, stride, padding, is_last)
    [-1,   8, (5, 5), (2, 2), (0, 0), False],
    (8,  16, (5, 5), (2, 2), (0, 0), False),
    (16,  32, (5, 5), (2, 2), (0, 0), False),
    (32,  64, (5, 5), (2, 2), (0, 0), False),
    (64, 128, (5, 5), (2, 2), (0, 0), False),
    (128, 128, (5, 5), (2, 2), (0, 0), True)
]

CNN_6_128_256 = [
    # (in_channels, out_channels, kernel_size, stride, padding, is_last)
    [-1,   8, (5, 7), (1, 3), (0, 0), False],
    (8,  16, (5, 7), (2, 2), (0, 0), False),
    (16,  32, (5, 7), (2, 2), (0, 0), False),
    (32,  64, (5, 7), (2, 2), (0, 0), False),
    (64, 128, (5, 7), (2, 2), (0, 0), False),
    (128, 128, (5, 5), (2, 2), (0, 0), True)
]

CNN_6_128_1024 = [
    # (in_channels, out_channels, kernel_size, stride, padding, is_last)
    [-1,   8, (7, 5), (3, 1), (0, 0), False],
    (8,  16, (7, 5), (2, 2), (0, 0), False),
    (16,  32, (7, 5), (2, 2), (0, 0), False),
    (32,  64, (7, 5), (2, 2), (0, 0), False),
    (64, 128, (7, 5), (2, 2), (0, 0), False),
    (128, 128, (5, 5), (2, 2), (0, 0), True)
]

CNN_6_128_768 = [
    # (in_channels, out_channels, kernel_size, stride, padding, is_last)
    [-1,   8, (7, 5), (3, 1), (0, 0), False],
    (8,  16, (7, 5), (2, 2), (0, 0), False),
    (16,  32, (5, 5), (2, 2), (0, 0), False),
    (32,  64, (5, 5), (2, 2), (0, 0), False),
    (64, 128, (5, 5), (2, 2), (0, 0), False),
    (128, 128, (5, 7), (2, 2), (0, 0), True)
]

CNN_6_128_384 = [
    # (in_channels, out_channels, kernel_size, stride, padding, is_last)
    [-1,   8, (5, 5), (1, 2), (0, 0), False],
    (8,  16, (5, 5), (2, 2), (0, 0), False),
    (16,  32, (5, 5), (2, 2), (0, 0), False),
    (32,  64, (5, 5), (2, 2), (0, 0), False),
    (64, 128, (7, 5), (2, 2), (0, 0), False),
    (128, 128, (7, 7), (2, 2), (0, 0), True)
]

CNN_6_128_128 = [
    # (in_channels, out_channels, kernel_size, stride, padding, is_last)
    [-1,   8, (3, 7), (1, 3), (0, 0), False],
    (8,  16, (3, 7), (1, 3), (0, 0), False),
    (16,  32, (5, 7), (2, 3), (0, 0), False),
    (32,  64, (5, 7), (2, 2), (0, 0), False),
    (64, 128, (5, 7), (2, 2), (0, 0), False),
    (128, 128, (5, 5), (2, 2), (0, 0), True)
]

CNN_6_128_1536 = [
    # (in_channels, out_channels, kernel_size, stride, padding, is_last)
    [-1,   8, (7, 5), (3, 2), (0, 0), False],
    (8,  16, (7, 5), (3, 2), (0, 0), False),
    (16,  32, (7, 3), (2, 2), (0, 0), False),
    (32,  64, (7, 3), (2, 2), (0, 0), False),
    (64, 128, (5, 3), (2, 1), (0, 0), False),
    (128, 128, (5, 3), (2, 1), (0, 0), True)
]

CNN_6_128_2048 = [
    # (in_channels, out_channels, kernel_size, stride, padding, is_last)
    [-1,   8, (7, 5), (3, 2), (0, 0), False],
    (8,  16, (7, 5), (3, 2), (0, 0), False),
    (16,  32, (7, 3), (3, 2), (0, 0), False),
    (32,  64, (7, 3), (2, 1), (0, 0), False),
    (64, 128, (5, 3), (2, 1), (0, 0), False),
    (128, 128, (5, 3), (2, 1), (0, 0), True)
]

# =================================================================================
# YNet architecture
# =================================================================================
# model_nlayer_lstm
YNet_6_128 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        [[0,  32, (5, 2), (2, 1), (1, 1)],
         (32,  64, (5, 2), (2, 1), (2, 1)),
         (64, 128, (5, 2), (2, 1), (2, 1))],
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        [(256,  64, (5, 2), (2, 1), (2, 0), (1, 0)),
         (128,  32, (5, 2), (2, 1), (2, 0), (1, 0)),
         [64,   0, (5, 2), (2, 1), (1, 0), (0, 0), True]]
}

# =================================================================================
# DCUNet architecture
# =================================================================================
# model_nlayer
DCUNet_10 = {
    # (in_channels, out_channels, kernel_size, stride, padding, causal)
    'encoder':
        ((1,  32, (7, 5), (2, 2)),
         (32, 64, (7, 5), (2, 2)),
         (64, 64, (5, 3), (2, 2)),
         (64, 64, (5, 3), (2, 1)),
         (64, 64, (5, 3), (2, 1))),
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        ((64, 64, (5, 3), (2, 1), (0, 1), (1, 0), False),
         (128, 64, (5, 3), (2, 2), (0, 1), (1, 0), False),
         (128, 64, (5, 3), (2, 2), (0, 1), (1, 0), False),
         (128, 32, (7, 5), (2, 2), (0, 1), (1, 0), False),
         (64,  1, (7, 5), (2, 2), (0, 1), (0, 1), True))
}

DCUNet_12 = {
    # (in_channels, out_channels, kernel_size, stride)
    'encoder':
        ((1,  64, (5, 5), (2, 2)),
         (64, 128, (5, 3), (2, 2)),
         (128, 128, (5, 3), (2, 2)),
         (128, 128, (5, 3), (2, 2)),
         (128, 128, (5, 3), (2, 2)),
         (128, 128, (3, 3), (2, 2))),
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        ((128, 128, (3, 3), (2, 2), (0, 0), (0, 0), False),
         (256, 128, (5, 3), (2, 2), (0, 0), (0, 0), False),
         (256, 128, (5, 3), (2, 2), (0, 0), (0, 0), False),
         (256, 128, (5, 3), (2, 2), (0, 0), (0, 0), False),
         (256,  64, (5, 3), (2, 2), (0, 0), (0, 0), False),
         (64,   1, (5, 5), (2, 2), (0, 0), (0, 0), True)),
    'shortPath': [False, True, True, True, True, False]
}

DCUNet_10_up = {
    # (in_channels, out_channels, kernel_size, stride)
    'encoder':
        ((32,  64, (3, 3), (2, 2), (0, 0)),
         (64, 128, (3, 3), (2, 2), (0, 0)),
         (128, 128, (3, 3), (2, 2), (0, 0)),
         (128, 128, (3, 3), (2, 2), (0, 0)),
         (128, 128, (3, 3), (2, 2), (0, 0))),
    # (in_channels, out_channels, kernel_size, stride, upSample, is_last)
    'decoder':
        ((128, 128, (3, 3), (1, 1), (0, 0), 3, False),
         (256, 128, (3, 3), (1, 1), (0, 0), 2, False),
         (256, 128, (3, 3), (1, 1), (0, 0), 2, False),
         (256,  64, (3, 3), (1, 1), (0, 0), 2, False),
         (128,   1, (3, 3), (1, 1), (0, 0), 2, True)),
    'shortPath': [False, True, True, True, True, True],
    'd_mode': 'up'
}

DCUNet_20 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        ((1, 32, (7, 1), (1, 1), (3, 0)),
         (32, 32, (1, 7), (1, 1), (0, 0)),
         (32, 64, (7, 5), (2, 2), (3, 0)),
         (64, 64, (7, 5), (2, 1), (3, 0)),
         (64, 64, (5, 3), (2, 2), (2, 0)),
         (64, 64, (5, 3), (2, 1), (2, 0)),
         (64, 64, (5, 3), (2, 2), (2, 0)),
         (64, 64, (5, 3), (2, 1), (2, 0)),
         (64, 64, (5, 3), (2, 2), (2, 0)),
         (64, 90, (5, 3), (2, 1), (2, 0))),
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        ((90, 64, (5, 3), (2, 1), (2, 0), (0, 0), False),
         (128, 64, (5, 3), (2, 2), (2, 0), (0, 0), False),
         (128, 64, (5, 3), (2, 1), (2, 0), (0, 0), False),
         (128, 64, (5, 3), (2, 2), (2, 0), (0, 0), False),
         (128, 64, (5, 3), (2, 1), (2, 0), (0, 0), False),
         (128, 64, (5, 3), (2, 2), (2, 0), (0, 0), False),
         (128, 64, (5, 3), (2, 1), (3, 0), (0, 0), False),
         (128, 32, (7, 5), (2, 2), (3, 0), (0, 0), False),
         (64, 32, (1, 7), (1, 1), (0, 0), (0, 0), False),
         (64,  1, (7, 1), (1, 1), (3, 0), (0, 0), True)),
    'shortPath': [False, True, True, True, True, True, True, True, True, True]
}

# =================================================================================
# Others architecture
# =================================================================================
Self2Self_12 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        ((1,  64, (3, 3), (2, 2), (0, 0)),
         (64, 128, (3, 3), (2, 2), (0, 0)),
         (128, 128, (3, 3), (2, 2), (0, 0)),
         (128, 128, (3, 3), (2, 2), (0, 0)),
         (128, 128, (3, 3), (2, 2), (0, 0)),
         (128, 128, (3, 3), (2, 2), (0, 0))),
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        ((128, 128, (3, 3), (2, 2), (0, 0), (0, 0), False),
         (256, 128, (3, 3), (2, 2), (0, 0), (0, 0), False),
         (256, 128, (3, 3), (2, 2), (0, 0), (0, 0), False),
         (256, 128, (3, 3), (2, 2), (0, 0), (0, 0), False),
         (256,  64, (3, 3), (2, 2), (0, 0), (0, 0), False),
         (128,   1, (3, 3), (2, 2), (0, 0), (0, 0), True)),
    'shortPath': [False, True, True, True, True, True],
}

DeConv_256 = ((1,  16, 5, 3),
              (16,  32, 5, 3),
              (32,  64, 5, 3),
              (64, 128, 5, 3),
              (128, 256, 5, 3))

DeConv_16 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        ((512, 256, 3),
         (256, 128, 3),
         (128,  64, 3),
         (64,  32, 3),
         (32,  16, 3)),
    'decoder':
        ((16,  32, 3),
         (64,  64, 3),
         (128, 128, 3),
         (256, 256, 3),
         (512, 512, 3, 1, 1, 0, True))
}

DeConv_512 = ((1,  32, 5, 3),
              (32,  64, 5, 3),
              (64, 128, 5, 3),
              (128, 256, 5, 3),
              (256, 512, 5, 3))

DeConv_1024 = ((1,  64, 5, 3),
               (64, 128, 5, 3),
               (128, 256, 5, 3),
               (256, 512, 5, 3),
               (512, 1024, 5, 3))

DeConv_2048 = ((1,  8, 5, 3),
               (8, 32, 5, 3),
               (32, 128, 5, 3),
               (128, 512, 5, 3),
               (512, 2048, 5, 3))


PhaseRecov_10_128 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        ((1,  64, 7, 1, 0),
         (64, 128, 5, 1, 0),
         (128, 128, 3, 1, 0),
         (128, 128, 3, 1, 0),
         (128, 128, 3, 1, 0)),
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        ((128, 128, 3, 1, 0, 0, False),
         (256, 128, 3, 1, 0, 0, False),
         (256, 128, 3, 1, 0, 0, False),
         (256,  64, 5, 1, 0, 0, False),
         (128,   1, 7, 1, 0, 0, True)),
    'shortPath': [False, True, True, True, True],
}

FNet_10_20 = {
    'encoder':
        [
            # ConvBlock: (in_channels, out_channels, kernel_size, stride, padding, causal)
            [-1, 10, (5, 2), (2, 1), (2, 1)],
            (10, 20, (3, 2), (1, 1), (1, 1)),
            # GTConvBlock: in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False
            (20, 20, (3, 3), (1, 1), (0, 1), (1, 1)),
            (20, 20, (3, 3), (1, 1), (0, 1), (2, 1)),
            (20, 20, (3, 3), (1, 1), (0, 1), (5, 1))
        ],
    'decoder':
        [
            # GTConvBlock: in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False
            (20, 20, (3, 3), (1, 1), (2*5, 1), (5, 1), True),
            (20, 20, (3, 3), (1, 1), (2*2, 1), (2, 1), True),
            (20, 20, (3, 3), (1, 1), (2*1, 1), (1, 1), True),
            # TransConvBlock: (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last, causal)
            (20, 10, (3, 2), (1, 1), (1, 0), (0, 0)),
            [10,  1, (5, 2), (2, 1), (2, 0), (1, 0), True]
        ],
    'n_ConvBlock': 2,
    'n_GTConvBlock': 3
}

FNet_10_16 = {
    'encoder':
        [
            # ConvBlock: (in_channels, out_channels, kernel_size, stride, padding, causal)
            [-1, 8, (5, 2), (2, 1), (0, 1)],
            (8, 16, (3, 2), (1, 1), (0, 1)),
            # GTConvBlock: in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False
            (16, 16, (3, 3), (1, 1), (0, 1), (1, 1)),
            (16, 16, (3, 3), (1, 1), (0, 1), (2, 1)),
            (16, 16, (3, 3), (1, 1), (0, 1), (5, 1))
        ],
    'decoder':
        [
            # GTConvBlock: in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False
            (16, 16, (3, 3), (1, 1), (2*5, 1), (5, 1), True),
            (16, 16, (3, 3), (1, 1), (2*2, 1), (2, 1), True),
            (16, 16, (3, 3), (1, 1), (2*1, 1), (1, 1), True),
            # TransConvBlock: (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last, causal)
            (16, 8, (3, 2), (1, 1), (0, 0), (0, 0)),
            [8, 1, (5, 2), (2, 1), (0, 0), (1, 0), True]
        ],
    'n_ConvBlock': 2,
    'n_GTConvBlock': 3
}

# reduce n_channel
FNet_10_12 = {
    'encoder':
        [
            # ConvBlock: (in_channels, out_channels, kernel_size, stride, padding, causal)
            [-1, 8, (5, 2), (2, 1), (0, 1)],
            (8, 12, (3, 2), (1, 1), (0, 1)),
            # GTConvBlock: in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False
            (12, 12, (3, 3), (1, 1), (0, 1), (1, 1)),
            (12, 12, (3, 3), (1, 1), (0, 1), (2, 1)),
            (12, 12, (3, 3), (1, 1), (0, 1), (5, 1))
        ],
    'decoder':
        [
            # GTConvBlock: in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False
            (12, 12, (3, 3), (1, 1), (2*5, 1), (5, 1), True),
            (12, 12, (3, 3), (1, 1), (2*2, 1), (2, 1), True),
            (12, 12, (3, 3), (1, 1), (2*1, 1), (1, 1), True),
            # TransConvBlock: (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last, causal)
            (12, 8, (3, 2), (1, 1), (0, 0), (0, 0)),
            [8, 1, (5, 2), (2, 1), (0, 0), (1, 0), True]
        ],
    'n_ConvBlock': 2,
    'n_GTConvBlock': 3
}

# reduce n_layer
FNet_8_16 = {
    'encoder':
        [
            # ConvBlock: (in_channels, out_channels, kernel_size, stride, padding, causal)
            [-1, 8, (5, 2), (2, 1), (0, 1)],
            (8, 16, (3, 2), (1, 1), (0, 1)),
            # GTConvBlock: in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False
            (16, 16, (3, 3), (1, 1), (0, 1), (1, 1)),
            (16, 16, (3, 3), (1, 1), (0, 1), (5, 1))
        ],
    'decoder':
        [
            # GTConvBlock: in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False
            (16, 16, (3, 3), (1, 1), (2*5, 1), (5, 1), True),
            (16, 16, (3, 3), (1, 1), (2*1, 1), (1, 1), True),
            # TransConvBlock: (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last, causal)
            (16, 8, (3, 2), (1, 1), (0, 0), (0, 0)),
            [8, 1, (5, 2), (2, 1), (0, 0), (1, 0), True]
        ],
    'n_ConvBlock': 2,
    'n_GTConvBlock': 2
}

LiteNet_12_20 = {
    'encoder':
        [
            # ConvBlock: (in_channels, out_channels, kernel_size, stride, padding, causal)
            [-1, 10, (5, 2), (2, 1), (2, 1)],
            (10, 20, (5, 2), (2, 1), (2, 1)),
            (20, 20, (5, 2), (2, 1), (2, 1)),
            # GTConvBlock: in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False
            (20, 20, (3, 3), (1, 1), (0, 1), (1, 1)),
            (20, 20, (3, 3), (1, 1), (0, 1), (2, 1)),
            (20, 20, (3, 3), (1, 1), (0, 1), (5, 1))
        ],
    'decoder':
        [
            # GTConvBlock: in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False
            (20, 20, (3, 3), (1, 1), (2*5, 1), (5, 1), True),
            (20, 20, (3, 3), (1, 1), (2*2, 1), (2, 1), True),
            (20, 20, (3, 3), (1, 1), (2*1, 1), (1, 1), True),
            # TransConvBlock: (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last, causal)
            (20, 20, (5, 2), (2, 1), (2, 0), (0, 0)),
            (20, 10, (5, 2), (2, 1), (2, 0), (0, 0)),
            [10,  1, (5, 2), (2, 1), (2, 0), (0, 0), True]
        ],
    'n_ConvBlock': 3,
    'n_GTConvBlock': 3
}

LiteNet_12_40 = {
    'encoder':
        [
            # ConvBlock: (in_channels, out_channels, kernel_size, stride, padding, causal)
            [-1, 10, (5, 2), (2, 1), (2, 1)],
            (10, 20, (5, 2), (2, 1), (2, 1)),
            (20, 40, (5, 2), (2, 1), (2, 1)),
            # GTConvBlock: in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False
            (40, 40, (3, 3), (1, 1), (0, 1), (1, 1)),
            (40, 40, (3, 3), (1, 1), (0, 1), (2, 1)),
            (40, 40, (3, 3), (1, 1), (0, 1), (5, 1))
        ],
    'decoder':
        [
            # GTConvBlock: in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False
            (40, 40, (3, 3), (1, 1), (2*5, 1), (5, 1), True),
            (40, 40, (3, 3), (1, 1), (2*2, 1), (2, 1), True),
            (40, 40, (3, 3), (1, 1), (2*1, 1), (1, 1), True),
            # TransConvBlock: (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last, causal)
            (40, 20, (5, 2), (2, 1), (2, 0), (0, 0)),
            (20, 10, (5, 2), (2, 1), (2, 0), (0, 0)),
            [10,  1, (5, 2), (2, 1), (2, 0), (0, 0), True]
        ],
    'n_ConvBlock': 3,
    'n_GTConvBlock': 3
}

LiteNet_10_32 = {
    'encoder':
        [
            # ConvBlock: (in_channels, out_channels, kernel_size, stride, padding, causal)
            [-1, 16, (5, 2), (3, 1), (0, 1)],
            (16, 32, (5, 2), (3, 1), (0, 1)),
            # GTConvBlock: in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False
            (32, 32, (3, 3), (1, 1), (0, 1), (1, 1)),
            (32, 32, (3, 3), (1, 1), (0, 1), (2, 1)),
            (32, 32, (3, 3), (1, 1), (0, 1), (5, 1))
        ],
    'decoder':
        [
            # GTConvBlock: in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False
            (32, 32, (3, 3), (1, 1), (2*5, 1), (5, 1), True),
            (32, 32, (3, 3), (1, 1), (2*2, 1), (2, 1), True),
            (32, 32, (3, 3), (1, 1), (2*1, 1), (1, 1), True),
            # TransConvBlock: (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last, causal)
            (32, 16, (5, 2), (3, 1), (0, 0), (2, 0)),
            [16,  1, (5, 2), (3, 1), (0, 0), (0, 0), True]
        ],
    'n_ConvBlock': 2,
    'n_GTConvBlock': 3,
    'n_freqs': 27
}

LiteNet_12_32 = {
    'encoder':
        [
            # ConvBlock: (in_channels, out_channels, kernel_size, stride, padding, causal)
            [-1,  8, (5, 2), (2, 1), (2, 1)],
            (8, 16, (5, 2), (2, 1), (2, 1)),
            (16, 32, (5, 2), (2, 1), (2, 1)),
            # GTConvBlock: in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False
            (32, 32, (3, 3), (1, 1), (0, 1), (1, 1)),
            (32, 32, (3, 3), (1, 1), (0, 1), (2, 1)),
            (32, 32, (3, 3), (1, 1), (0, 1), (5, 1))
        ],
    'decoder':
        [
            # GTConvBlock: in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False
            (32, 32, (3, 3), (1, 1), (2*5, 1), (5, 1), True),
            (32, 32, (3, 3), (1, 1), (2*2, 1), (2, 1), True),
            (32, 32, (3, 3), (1, 1), (2*1, 1), (1, 1), True),
            # TransConvBlock: (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last, causal)
            (32, 16, (5, 2), (2, 1), (2, 0), (0, 0)),
            (16,  8, (5, 2), (2, 1), (2, 0), (0, 0)),
            [8,  1, (5, 2), (2, 1), (2, 0), (0, 0), True]
        ],
    'n_ConvBlock': 3,
    'n_GTConvBlock': 3
}

# =================================================================================
# Alias
# =================================================================================

TNet_6_64 = UNet_6_64

TNet_6_128 = UNet_6_128

TNet_8_256 = UNet_8_256

DNet_6_64 = UNet_6_64

DNet_6_128 = UNet_6_128

DNet_8_256 = UNet_8_256

DNet_6_32 = UNet_6_32

DNet_6_16 = UNet_6_16