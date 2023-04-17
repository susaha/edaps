_base_ = ['bottomup_sepaspp_mitb5_core.py']
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    decode_head=dict(
    type='DAFormerHeadPanopticShared',
        decoder_params=dict(
            fusion_cfg=dict(
                _delete_=True,
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg))))
