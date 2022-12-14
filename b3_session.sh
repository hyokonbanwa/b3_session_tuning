#!/bin/sh
python b3_session_cnn.py name=pyfile seed=0 epochs=20 transform.pattern=4 dataloader.batch_size=512 model.enc_dim=256 optimizer.lr=1.0e-4
