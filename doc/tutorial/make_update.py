#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
# This is from the SES3D Python tools.
import models as m

# Input. ----------------------------------------------------------------------
# Directory where the current model is located.
dir_current_model = "../MODELS/MODELS_3D/MODEL.66/"

# Directory where test models will be written.
dir_test_models = "../MODELS/MODELS_3D/MODEL.67/"

# Directory where current gradients are located.
dir_kernels = "../GRADIENTS/ITERATION_1/"

# List of events for which kernels are available.
eventlist = [
    "GCMT_event_NORTHERN_ITALY_Mag_4.9_2000-8-21-17",
    "GCMT_event_NORTHWESTERN_BALKAN_REGION_Mag_5.9_1980-5-18-20"]

# Smoothing iterations.
sigma = 3.0

# Test step lengths.
gamma = [0.15, 0.5, 1.0]

# Read current model. ---------------------------------------------------------
print "read current model from directory " + dir_current_model

csv = m.ses3d_model()
csh = m.ses3d_model()
rho = m.ses3d_model()
cp = m.ses3d_model()

csv.read(dir_current_model, 'dvsv')
csh.read(dir_current_model, 'dvsh')
rho.read(dir_current_model, 'drho')
cp.read(dir_current_model, 'dvp')

# Read gradients of the current iteration. ------------------------------------
# Initialisation.
grad_csv = m.ses3d_model()
grad_csh = m.ses3d_model()
grad_rho = m.ses3d_model()
grad_cp = m.ses3d_model()

grad_csv_k = m.ses3d_model()
grad_csh_k = m.ses3d_model()
grad_rho_k = m.ses3d_model()
grad_cp_k = m.ses3d_model()

# Loop through events.
for k in eventlist:

    print "read kernels for event " + str(k)

    if k == eventlist[0]:

        #  Read kernels.
        grad_csv.read(dir_kernels + str(k) + '/', 'gradient_csv')
        grad_csh.read(dir_kernels + str(k) + '/', 'gradient_csh')
        grad_rho.read(dir_kernels + str(k) + '/', 'gradient_rho')
        grad_cp.read(dir_kernels + str(k) + '/', 'gradient_cp')

        #  Clip the upper percentile to remove singularities.
        grad_csv.clip_percentile(99.9)
        grad_csh.clip_percentile(99.9)
        grad_cp.clip_percentile(99.9)
        grad_rho.clip_percentile(99.9)

    else:

        # Read kernels.
        grad_csv_k.read(dir_kernels + str(k) + '/', 'gradient_csv')
        grad_csh_k.read(dir_kernels + str(k) + '/', 'gradient_csh')
        grad_rho_k.read(dir_kernels + str(k) + '/', 'gradient_rho')
        grad_cp_k.read(dir_kernels + str(k) + '/', 'gradient_cp')

        # Clip the upper percentile to remove singularities.
        grad_csv_k.clip_percentile(99.9)
        grad_csh_k.clip_percentile(99.9)
        grad_cp_k.clip_percentile(99.9)
        grad_rho_k.clip_percentile(99.9)

        # Add to the previous kernels.
        grad_csv = grad_csv + grad_csv_k
        grad_csh = grad_csh + grad_csh_k
        grad_rho = grad_rho + grad_rho_k
        grad_cp = grad_cp + grad_cp_k

# Compute the new descent direction. ------------------------------------------
# Simple gradient descent.
h_csv = -1.0 * grad_csv
h_csh = -1.0 * grad_csh
h_cp = -1.0 * grad_cp
h_rho = -1.0 * grad_rho

# Store new gradients and descent directions. ---------------------------------
grad_csv.write(dir_kernels, 'gradient_csv_sum')
grad_csh.write(dir_kernels, 'gradient_csh_sum')
grad_rho.write(dir_kernels, 'gradient_rho_sum')
grad_cp.write(dir_kernels, 'gradient_cp_sum')

h_csv.write(dir_kernels, 'h_csv_sum')
h_csh.write(dir_kernels, 'h_csh_sum')
h_rho.write(dir_kernels, 'h_rho_sum')
h_cp.write(dir_kernels, 'h_cp_sum')

# Apply smoothing. ------------------------------------------------------------
print "smoothing"

h_csv.smooth_horizontal(sigma, filter_type='neighbour')
h_csh.smooth_horizontal(sigma, filter_type='neighbour')
h_cp.smooth_horizontal(sigma, filter_type='neighbour')
h_rho.smooth_horizontal(sigma, filter_type='neighbour')

# Relative importance of the kernels. -----------------------------------------
sum_csv = 0.0
sum_csh = 0.0
sum_rho = 0.0
sum_cp = 0.0

for n in range(grad_csv.nsubvol):

    sum_csv = sum_csv + np.sum(np.abs(h_csv.m[n].v))
    sum_csh = sum_csh + np.sum(np.abs(h_csh.m[n].v))
    sum_rho = sum_rho + np.sum(np.abs(h_rho.m[n].v))
    sum_cp = sum_cp + np.sum(np.abs(h_cp.m[n].v))

max_sum = np.max([sum_csv, sum_csh, sum_cp, sum_rho])

scale_csv_total = sum_csv / max_sum
scale_csh_total = sum_csh / max_sum
scale_rho_total = sum_rho / max_sum
scale_cp_total = sum_cp / max_sum

print scale_csv_total, scale_csh_total, scale_rho_total, scale_cp_total

# Depth scaling.---------------------------------------------------------------
max_csv = []
max_csh = []
max_rho = []
max_cp = []

for n in range(h_csv.nsubvol):

    max_csv.append(np.max(np.abs(h_csv.m[n].v[:, :, :])))
    max_csh.append(np.max(np.abs(h_csh.m[n].v[:, :, :])))
    max_rho.append(np.max(np.abs(h_rho.m[n].v[:, :, :])))
    max_cp.append(np.max(np.abs(h_cp.m[n].v[:, :, :])))

max_csv = np.max(max_csv)
max_csh = np.max(max_csh)
max_rho = np.max(max_rho)
max_cp = np.max(max_cp)

damp = 0.3

for n in range(h_csv.nsubvol):

    for k in range(len(h_csv.m[n].r) - 1):

        h_csv.m[n].v[:, :, k] = h_csv.m[n].v[:, :, k] / (
            damp * max_csv + np.max(np.abs(h_csv.m[n].v[:, :, k])))
        h_csh.m[n].v[:, :, k] = h_csh.m[n].v[:, :, k] / (
            damp * max_csh + np.max(np.abs(h_csh.m[n].v[:, :, k])))
        h_rho.m[n].v[:, :, k] = h_rho.m[n].v[:, :, k] / (
            damp * max_rho + np.max(np.abs(h_rho.m[n].v[:, :, k])))
        h_cp.m[n].v[:, :, k] = h_cp.m[n].v[:, :, k] / (
            damp * max_cp + np.max(np.abs(h_cp.m[n].v[:, :, k])))

h_csv = scale_csv_total * h_csv
h_csh = scale_csh_total * h_csh
h_rho = scale_rho_total * h_rho
h_cp = scale_cp_total * h_cp

# Store the smoothed descent directions. --------------------------------------
h_csv.write(dir_kernels, 'h_csv_sum_smoothed')
h_csh.write(dir_kernels, 'h_csh_sum_smoothed')
h_rho.write(dir_kernels, 'h_rho_sum_smoothed')
h_cp.write(dir_kernels, 'h_cp_sum_smoothed')

# Compute and store updates. --------------------------------------------------
for k in np.arange(len(gamma)):
    csv_new = csv + gamma[k] * h_csv
    csh_new = csh + gamma[k] * h_csh
    rho_new = rho + gamma[k] * h_rho
    cp_new = cp + gamma[k] * h_cp

    csv_new.write(dir_test_models, 'dvsv.' + str(gamma[k]))
    csh_new.write(dir_test_models, 'dvsh.' + str(gamma[k]))
    rho_new.write(dir_test_models, 'drho.' + str(gamma[k]))
    cp_new.write(dir_test_models, 'dvp.' + str(gamma[k]))
