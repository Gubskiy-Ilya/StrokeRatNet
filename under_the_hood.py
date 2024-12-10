import os, sys, io, cv2, gc
import torch
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch, ThreadDataLoader, set_track_meta
from monai.networks.nets import SwinUNETR
from monai.utils import first
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureType,
    EnsureTyped,
    EnsureChannelFirstd,
    ThresholdIntensityd,
    CenterSpatialCropd,
    RandRotated,
    RandGaussianNoised,
    RandAffined,
    RandCropByPosNegLabeld,
    ToDeviced,
)
sitk.ProcessObject.SetGlobalDefaultThreader('Pool')

def dcm_to_nii_n4(dcmfol):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcmfol)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    image = sitk.Cast(image, sitk.sitkFloat32)
    image = n4bias(image)
    sitk.WriteImage(image, f"temp.nii")

def n4bias(inputImage , shrinkFactor=4 , ControlPoints=[5,5,5], Threshold=0.0001, Iterations=[50,50,50]):
    image = sitk.Shrink(inputImage, [shrinkFactor] * inputImage.GetDimension())
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetNumberOfControlPoints(ControlPoints)
    corrector.SetConvergenceThreshold(Threshold)
    corrector.SetMaximumNumberOfIterations(Iterations)
    OutputImageShrink = corrector.Execute(image)
    log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
    corrected_image_full_resolution = inputImage / sitk.Exp(log_bias_field)
    return corrected_image_full_resolution

def stroke_volume():
    data_dicts = [{"image": os.path.abspath('temp.nii')}]
    device = torch.device("cuda:0")
    model = SwinUNETR(
        img_size=(64, 64, 64), 
        in_channels=1, 
        out_channels=2, 
        feature_size=24,
        normalize=True,
        use_checkpoint=True,
        use_v2=True,
    ).to(device)
    model.load_state_dict(torch.load('srn.pth', map_location=device, weights_only=False))
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    test_transforms = Compose(
        [
            LoadImaged(reader='NibabelReader', keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            EnsureTyped(keys=["image"]),
            ToDeviced(keys=["image"], device=f"cuda:0"),
            Orientationd(keys=["image"], axcodes='RSA'),
            Spacingd(
                keys=["image"],
                pixdim=(0.2, 0.2, 0.2),
                mode=("bilinear"),
            ),
            CenterSpatialCropd(keys=["image"], roi_size=[105, 195, 64]),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)
        ]
    )
    
    test_ds = CacheDataset(data=data_dicts, transform=test_transforms, cache_num=10, cache_rate=1.0, num_workers=2)
    test_loader = ThreadDataLoader(test_ds, num_workers=0, batch_size=10)
    model.eval()
    with torch.no_grad():
        img = test_ds[0]["image"]
        val_inputs = torch.unsqueeze(img, 1).to(device)
        with torch.amp.autocast("cuda"):
            val_outputs = sliding_window_inference(val_inputs, (64, 64, 64), 32, model, overlap=0.7)

    t_img = img.squeeze().cpu().detach().numpy()
    t_out_pred = post_pred(val_outputs.squeeze())
    t_out = t_out_pred.squeeze().cpu().detach().numpy()[1,...]
    img_f = sitk.GetImageFromArray(t_img)
    lbl_f = sitk.GetImageFromArray(t_out)
    sitk.WriteImage(img_f, f"./results/image.nii")
    sitk.WriteImage(lbl_f, f"./results/label.nii")
    if os.path.exists("temp.nii"):
        os.remove("temp.nii")
    
    t_img = img.squeeze().cpu().detach().numpy().transpose([1,2,0])
    t_img = np.flip(t_img, (0, 1, 2))
    t_out_pred = post_pred(val_outputs.squeeze())
    t_out = t_out_pred.squeeze().cpu().detach().numpy()[1,...].transpose([1,2,0])
    t_out = np.flip(t_out, (0, 1, 2))
    
    fig, axs = plt.subplots(2, 3, figsize=(11, 10), constrained_layout=True, gridspec_kw={'width_ratios': [2.5, 1, .6]})
    fig.set_facecolor("black")
    fig.suptitle(f'Stroke volume {"%.2f" % (t_out.sum()*.2*.2*.2)} mm3', fontsize=32, color="white")
    if t_out.sum() == 0:
        x_cord = t_out.shape[0]//2
        y_cord = t_out.shape[1]//2
        z_cord = t_out.shape[2]//2
    else:
        x_cord = np.argmax(t_out.sum(axis=(1,2)))
        y_cord = np.argmax(t_out.sum(axis=(0,2)))
        z_cord = np.argmax(t_out.sum(axis=(0,1)))
    w_min = np.mean(t_img[t_img<0])
    w_max = np.mean(t_img[t_img>0])*4
    axs[0, 0].imshow(t_img[x_cord,:,:], vmin=w_min, vmax=w_max, cmap='Greys_r', interpolation='bilinear')
    axs[0, 0].axis('off')
    axs[0, 1].imshow(t_img[:,y_cord,:], vmin=w_min, vmax=w_max, cmap='Greys_r', interpolation='bilinear')
    axs[0, 1].axis('off')
    axs[0, 2].imshow(t_img[:,:,z_cord], vmin=w_min, vmax=w_max, cmap='Greys_r', interpolation='bilinear')
    axs[0, 2].axis('off')
    axs[1, 0].imshow(t_img[x_cord,:,:], vmin=w_min, vmax=w_max, cmap='Greys_r', alpha=1, interpolation='bilinear')
    axs[1, 0].imshow(t_out[x_cord,:,:], vmin=0, vmax=1, cmap='Reds', alpha=np.float64(t_out[x_cord,:,:]), interpolation='none')
    axs[1, 0].axis('off')
    axs[1, 1].imshow(t_img[:,y_cord,:], vmin=w_min, vmax=w_max, cmap='Greys_r', alpha=0.7, interpolation='bilinear')
    axs[1, 1].imshow(t_out[:,y_cord,:], vmin=0, vmax=1, cmap='Reds', alpha=np.float64(t_out[:,y_cord,:]), interpolation='none')
    axs[1, 1].axis('off')
    axs[1, 2].imshow(t_img[:,:,z_cord], vmin=w_min, vmax=w_max, cmap='Greys_r', alpha=0.7, interpolation='bilinear')
    axs[1, 2].imshow(t_out[:,:,z_cord], vmin=0, vmax=1, cmap='Reds', alpha=np.float64(t_out[:,:,z_cord]), interpolation='none')
    axs[1, 2].axis('off')
    plt.tight_layout()
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight', transparent=False, facecolor='black')
    buf.seek(0) 
    img = Image.open(buf)
    img.save(f"./results/segmentation.png")
    torch.cuda.empty_cache()
    gc.collect()
    print(f'Stroke volume {"%.2f" % (t_out.sum()*.2*.2*.2)} mm3')

def registration_sitk(fixed, moving, trans, metric, optim, shrink, smooth, interp, HistogramBins=50):
    if trans == "Euler3D":
        initial_transform = sitk.CenteredTransformInitializer(fixed,
                                                              moving,
                                                              sitk.Euler3DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)
    elif trans == "Affine3D":
        initial_transform = sitk.CenteredTransformInitializer(fixed,
                                                              moving,
                                                              sitk.AffineTransform(3),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = sitk.ImageRegistrationMethod()
    if metric == "MU":
        registration_method.SetMetricAsMattesMutualInformation(
            numberOfHistogramBins=HistogramBins)
    elif metric == "MS":
        registration_method.SetMetricAsMeanSquares()
    elif metric == "ANTS":
        registration_method.SetMetricAsANTSNeighborhoodCorrelation(radius=3)
    elif metric == "COR":
        registration_method.SetMetricAsCorrelation()
    elif metric == "MASK":
        registration_method.SetMetricFixedMask(fixed)

    if optim == "LBFG":
        registration_method.SetOptimizerAsLBFGSB()
    elif optim == "GD":
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)

    if shrink != None:
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=shrink)
    if smooth != None:
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=smooth)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.3)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(sitk.Cast(fixed, sitk.sitkFloat32),
                                                  sitk.Cast(moving, sitk.sitkFloat32))

    if interp == "linear":
        moving_ex = sitk.Resample(moving, fixed, final_transform,
                                  sitk.sitkLinear, 0.0, fixed.GetPixelID())
    if interp == "nearest":
        moving_ex = sitk.Resample(moving, fixed, final_transform,
                                  sitk.sitkNearestNeighbor, 0.0, fixed.GetPixelID())

    return moving_ex, final_transform

def set_dim(img, img_orig):
    img.SetOrigin(img_orig.GetOrigin())
    img.SetSpacing(img_orig.GetSpacing())
    img.SetDirection(img_orig.GetDirection())
    return img

def registration_step(image0, mask0, image_atlas, var_par):
    image_0_n, tfm1 = registration_sitk(image_atlas,  image0,
                                 "Euler3D", var_par[0], var_par[1], [2, 1], [2, 1], 
                                 "linear", 50)
    image_0_n, tfm2 = registration_sitk(image_atlas, image_0_n,
                                 "Affine3D", var_par[0], var_par[1], [2, 1], [2, 1], 
                                 "linear", 50)
    mask0_n = sitk.Resample(mask0, image_atlas, tfm1, sitk.sitkNearestNeighbor, 0.0, image_atlas.GetPixelID())
    mask0_n = sitk.Resample(mask0_n, image_atlas, tfm2, sitk.sitkNearestNeighbor, 0.0, image_atlas.GetPixelID())
    return mask0_n, image_0_n

def ica_atlas(variation):
    image_atlas = sitk.ReadImage('template.nii', sitk.sitkFloat32)
    image0 = sitk.ReadImage('./results/image.nii', sitk.sitkFloat32)
    mask0 = sitk.ReadImage('./results/label.nii', sitk.sitkFloat32)

    if variation==1:
        var_par = ["MU", "LBFG"]
    elif variation==2:
        var_par = ["MU", "GD"]
    elif variation==3:
        var_par = ["MS", "GD"]
    else:
        var_par = ["MS", "LBFG"]
    
    for n in tqdm(range(3)):
        try:
            mask0_n, image_0_n = registration_step(image0, mask0, image_atlas, var_par)
            set_dim(mask0_n, image_atlas)
            sitk.WriteImage(mask0_n, f"./results/label_atlas.nii")
        except:
            continue
        else:
            break
            
    fixed_array = sitk.GetArrayFromImage(image_atlas)
    fixed_array = fixed_array.transpose([1,2,0])
    fixed_array = np.flip(fixed_array, (0, 1, 2))
    
    moving_array = sitk.GetArrayFromImage(image_0_n)
    moving_array = moving_array.transpose([1,2,0])
    moving_array = np.flip(moving_array, (0, 1, 2))
    
    fig, axs = plt.subplots(1, 3, figsize=(6, 3))
    fig.set_tight_layout('pad')
    plt.subplots_adjust(wspace=0, hspace=0)
    axs[0].imshow(moving_array[:, :, moving_array.shape[2]//2], vmin=1, vmax=2, cmap="Reds")
    axs[0].imshow(fixed_array[:, :, fixed_array.shape[2]//2], vmin=0.2, vmax=2, cmap="Greys_r", alpha=0.6)
    axs[0].axis('off')
    axs[1].imshow(moving_array[:, moving_array.shape[1]//2, :], vmin=1, vmax=2, cmap="Reds")
    axs[1].imshow(fixed_array[:, fixed_array.shape[1]//2, :], vmin=0.2, vmax=2, cmap="Greys_r", alpha=0.6)
    axs[1].axis('off')
    axs[2].imshow(moving_array[moving_array.shape[0]//2, :, :], vmin=1, vmax=2, cmap="Reds")
    axs[2].imshow(fixed_array[fixed_array.shape[0]//2, :, :], vmin=0.2, vmax=2, cmap="Greys_r", alpha=0.6)
    axs[2].axis('off')
    fig.set_facecolor("Black")

def ica_blood_supply():
    image_ica_supply = sitk.ReadImage('ica_atlas.nii', sitk.sitkFloat32)
    stroke_arrea = sitk.ReadImage('./results/label_atlas.nii', sitk.sitkFloat32)
    image_ica_supply_a = sitk.GetArrayFromImage(image_ica_supply)
    image_ica_supply_a = image_ica_supply_a.transpose([1,2,0])
    image_ica_supply_a = np.flip(image_ica_supply_a, (0, 1, 2))
    stroke_arrea_a = sitk.GetArrayFromImage(stroke_arrea)
    stroke_arrea_a = stroke_arrea_a.transpose([1,2,0])
    stroke_arrea_a = np.flip(stroke_arrea_a, (0, 1, 2))
    
    a1 = np.sum(image_ica_supply_a[stroke_arrea_a==1]==1) / np.sum(image_ica_supply_a==1)
    a2 = np.sum(image_ica_supply_a[stroke_arrea_a==1]==2) / np.sum(image_ica_supply_a==2)
    a3 = np.sum(image_ica_supply_a[stroke_arrea_a==1]==3) / np.sum(image_ica_supply_a==3)
    a4 = np.sum(image_ica_supply_a[stroke_arrea_a==1]==4) / np.sum(image_ica_supply_a==4)
    if a1 > 1: a1 = 1
    if a2 > 1: a2 = 1
    if a3 > 1: a3 = 1
    if a4 > 1: a4 = 1
    print(f'Cortical MCA {"%.1f" % (a4*100)}%')
    print(f'Subcortical MCA {"%.1f" % (a2*100)}%')
    print(f'AChA {"%.1f" % (a1*100)}%')
    print(f'HTA {"%.1f" % (a3*100)}%')
    
    fig, axs = plt.subplots(4, 3, figsize=(4, 8), constrained_layout=True, gridspec_kw={'width_ratios': [1.25, 1, .6]})
    fig.set_facecolor("black")
    axs[0, 0].set_title(f'Cortical MCA = {"%.1f" % (a4*100)}%', fontsize=12, color="white")
    axs[0, 0].imshow(np.where(image_ica_supply_a==4, 1, 0)[78, :, 44:], vmin=0, vmax=1, cmap='Reds', interpolation='bilinear')
    axs[0, 0].imshow(stroke_arrea_a[78,:,44:], vmin=0, vmax=1.4, cmap='Greys_r', alpha=0.5, interpolation='bilinear')
    axs[0, 0].axis('off')
    axs[0, 1].imshow(np.where(image_ica_supply_a==4, 1, 0)[:,49,:], vmin=0, vmax=1, cmap='Reds', interpolation='bilinear')
    axs[0, 1].imshow(stroke_arrea_a[:,49,:], vmin=0, vmax=1.4, cmap='Greys_r', alpha=0.5, interpolation='bilinear')
    axs[0, 1].axis('off')
    axs[0, 2].imshow(np.where(image_ica_supply_a==4, 1, 0)[:,:,80], vmin=0, vmax=1, cmap='Reds', interpolation='bilinear')
    axs[0, 2].imshow(stroke_arrea_a[:,:,80], vmin=0, vmax=1.4, cmap='Greys_r', alpha=0.5, interpolation='bilinear')
    axs[0, 2].axis('off')
    axs[1, 0].set_title(f'Subcortical MCA = {"%.1f" % (a2*100)}%', fontsize=12, color="white")
    axs[1, 0].imshow(np.where(image_ica_supply_a==2, 1, 0)[72, :, 44:], vmin=0, vmax=1, cmap='Oranges', interpolation='bilinear')
    axs[1, 0].imshow(stroke_arrea_a[72,:,44:], vmin=0, vmax=1.4, cmap='Greys_r', alpha=0.4, interpolation='bilinear')
    axs[1, 0].axis('off')
    axs[1, 1].imshow(np.where(image_ica_supply_a==2, 1, 0)[:,33,:], vmin=0, vmax=1, cmap='Oranges', interpolation='bilinear')
    axs[1, 1].imshow(stroke_arrea_a[:,33,:], vmin=0, vmax=1.4, cmap='Greys_r', alpha=0.4, interpolation='bilinear')
    axs[1, 1].axis('off')
    axs[1, 2].imshow(np.where(image_ica_supply_a==2, 1, 0)[:,:,66], vmin=0, vmax=1, cmap='Oranges', interpolation='bilinear')
    axs[1, 2].imshow(stroke_arrea_a[:,:,66], vmin=0, vmax=1.4, cmap='Greys_r', alpha=0.4, interpolation='bilinear')
    axs[1, 2].axis('off')
    axs[2, 0].set_title(f'AChA = {"%.1f" % (a1*100)}%', fontsize=12, color="white")
    axs[2, 0].imshow(np.where(image_ica_supply_a==1, 1, 0)[81, :, 44:], vmin=0, vmax=1, cmap='Greens', interpolation='bilinear')
    axs[2, 0].imshow(stroke_arrea_a[81,:,44:], vmin=0, vmax=1.4, cmap='Greys_r', alpha=0.4, interpolation='bilinear')
    axs[2, 0].axis('off')
    axs[2, 1].imshow(np.where(image_ica_supply_a==1, 1, 0)[:,42,:], vmin=0, vmax=1, cmap='Greens', interpolation='bilinear')
    axs[2, 1].imshow(stroke_arrea_a[:,42,:], vmin=0, vmax=1.4, cmap='Greys_r', alpha=0.4, interpolation='bilinear')
    axs[2, 1].axis('off')
    axs[2, 2].imshow(np.where(image_ica_supply_a==1, 1, 0)[:,:,81], vmin=0, vmax=1, cmap='Greens', interpolation='bilinear')
    axs[2, 2].imshow(stroke_arrea_a[:,:,81], vmin=0, vmax=1.4, cmap='Greys_r', alpha=0.4, interpolation='bilinear')
    axs[2, 2].axis('off')
    axs[3, 0].set_title(f'HTA = {"%.1f" % (a3*100)}%', fontsize=12, color="white")
    axs[3, 0].imshow(np.where(image_ica_supply_a==3, 1, 0)[86, :, 44:], vmin=0, vmax=1, cmap='Blues', interpolation='bilinear')
    axs[3, 0].imshow(stroke_arrea_a[86,:,44:], vmin=0, vmax=1.4, cmap='Greys_r', alpha=0.4, interpolation='bilinear')
    axs[3, 0].axis('off')
    axs[3, 1].imshow(np.where(image_ica_supply_a==3, 1, 0)[:,46,:], vmin=0, vmax=1, cmap='Blues', interpolation='bilinear')
    axs[3, 1].imshow(stroke_arrea_a[:,46,:], vmin=0, vmax=1.4, cmap='Greys_r', alpha=0.4, interpolation='bilinear')
    axs[3, 1].axis('off')
    axs[3, 2].imshow(np.where(image_ica_supply_a==3, 1, 0)[:,:,63], vmin=0, vmax=1, cmap='Blues', interpolation='bilinear')
    axs[3, 2].imshow(stroke_arrea_a[:,:,63], vmin=0, vmax=1.4, cmap='Greys_r', alpha=0.4, interpolation='bilinear')
    axs[3, 2].axis('off')
    plt.tight_layout()
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight', transparent=False, facecolor='black')
    buf.seek(0) 
    img = Image.open(buf)
    img.save(f"./results/ica_blood_supply.png")
    gc.collect()
