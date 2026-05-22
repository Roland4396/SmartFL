$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Tag = "vit_hidden_fullcut_v1"
$ResultsDir = Join-Path $Root "tdd_batch_results\$Tag"
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

function Run-Step {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][string[]]$ArgsList
    )

    $LogPath = Join-Path $ResultsDir "$Name.log"
    $Start = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$Start] START $Name" | Tee-Object -FilePath $LogPath
    "conda run -n scalefl python $($ArgsList -join ' ')" | Tee-Object -FilePath $LogPath -Append

    & conda run -n scalefl python @ArgsList 2>&1 | Tee-Object -FilePath $LogPath -Append
    if ($LASTEXITCODE -ne 0) {
        $End = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        "[$End] FAILED $Name exit=$LASTEXITCODE" | Tee-Object -FilePath $LogPath -Append
        exit $LASTEXITCODE
    }

    $End = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$End] DONE $Name" | Tee-Object -FilePath $LogPath -Append
}

$CommonPrep = @(
    "main.py",
    "--arch", "vit_small_4",
    "--model", "vit",
    "--seed", "0",
    "--gpu_idx", "0",
    "--use_gpu", "1",
    "--num_architectures", "50000",
    "--episodes_per_batch", "100",
    "--stages_only", "1,2"
)

Run-Step "prep_cifar10_stage1_2" ($CommonPrep + @(
    "--data", "cifar10",
    "--supernet_save_path", "vit_small_cifar10_supernet.pth",
    "--config_library_path", "vit_small_cifar10_architecture_library.json"
))

Run-Step "prep_cifar100_stage1_2" ($CommonPrep + @(
    "--data", "cifar100",
    "--supernet_save_path", "vit_small_cifar100_supernet.pth",
    "--config_library_path", "vit_small_cifar100_architecture_library.json"
))

Run-Step "prep_tiny_imagenet_stage1_2" ($CommonPrep + @(
    "--data", "tiny_imagenet",
    "--supernet_save_path", "vit_small_tiny_imagenet_supernet.pth",
    "--config_library_path", "vit_small_tiny_imagenet_architecture_library.json"
))

Run-Step "stage3_compare_sweep" @(
    "run_tdd_batch.py",
    "--compare_sweep",
    "--archs", "vit_small_4",
    "--datasets", "cifar10", "cifar100", "tiny_imagenet",
    "--alphas", "1", "100",
    "--tdd_modes", "on", "off",
    "--client_split_ratios", "4", "3", "2", "1",
    "--num_rounds", "400",
    "--validate_every", "100",
    "--seed", "0",
    "--gpu_idx", "0",
    "--use_gpu", "1",
    "--num_architectures", "50000",
    "--episodes_per_batch", "100",
    "--run_tag", $Tag,
    "--results_dir", "tdd_batch_results\$Tag",
    "--max_parallel", "1",
    "--rerun",
    "--no_auto_resume"
)
