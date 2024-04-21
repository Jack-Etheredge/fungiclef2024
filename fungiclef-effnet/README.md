## Performance notes:

- openGANfea variations vs softmax thresholding:
    - discriminator only:
        - best discriminator: seesaw_04_02_2024_dlr_1e-6_open1closed0_no_gen/epoch-11-discriminator.pth
            - selection roc-auc 0.6117849906483791 and evaluation roc-auc 0.5604385963920865
    - GAN open 1 closed 0:
        - best discriminator: seesaw_04_02_2024_dlr_1e-6_open1closed0/epoch-10-discriminator.pth
            - selection roc-auc 0.6020331982543641 and evaluation roc-auc 0.5775920077303734
    - GAN open 0 closed 1:
        - best discriminator: seesaw_04_02_2024_dlr_1e-6_glr_1e-6_open0closed1/epoch-20-discriminator.pth
            - selection roc-auc 0.6231694201995013 and evaluation roc-auc 0.6029468184269686
    - softmax thresholding (overly optimistic; set threshold on the test set):
        - roc-auc 0.5447452440605403
    - temp scaled softmax thresholding (overly optimistic; set threshold on the test set):
        - roc-auc 0.544770863626587