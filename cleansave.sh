#!/usr/bin/env bash

find saved_models -type f -name "latest_model.pth" -delete
zip -r saved_models.zip saved_models/