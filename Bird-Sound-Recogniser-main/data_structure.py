import os
from collections import defaultdict


def show_directory_tree(root_path, max_files_per_folder=5, max_depth=10):
    """
    展示目录树结构

    Args:
        root_path: 根目录路径
        max_files_per_folder: 每个文件夹最多显示的文件数量
        max_depth: 最大显示深度
    """

    def _get_tree_structure(path, prefix="", depth=0):
        if depth > max_depth:
            return []

        if not os.path.exists(path):
            return [f"{prefix}❌ 路径不存在: {path}"]

        items = []
        try:
            entries = sorted(os.listdir(path))
            dirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]
            files = [e for e in entries if os.path.isfile(os.path.join(path, e))]

            # 显示目录
            for i, dir_name in enumerate(dirs):
                is_last_dir = (i == len(dirs) - 1) and len(files) == 0
                current_prefix = "└── " if is_last_dir else "├── "
                items.append(f"{prefix}{current_prefix}📁 {dir_name}/")

                # 递归显示子目录
                next_prefix = prefix + ("    " if is_last_dir else "│   ")
                sub_items = _get_tree_structure(
                    os.path.join(path, dir_name),
                    next_prefix,
                    depth + 1
                )
                items.extend(sub_items)

            # 显示文件
            displayed_files = files[:max_files_per_folder]
            for i, file_name in enumerate(displayed_files):
                is_last = i == len(displayed_files) - 1
                current_prefix = "└── " if is_last else "├── "

                # 根据文件类型显示不同图标
                if file_name.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    icon = "🎵"
                elif file_name.lower().endswith(('.json', '.txt')):
                    icon = "📄"
                elif file_name.lower().endswith(('.py', '.ipynb')):
                    icon = "🐍"
                else:
                    icon = "📄"

                items.append(f"{prefix}{current_prefix}{icon} {file_name}")

            # 如果有更多文件没显示，添加提示
            if len(files) > max_files_per_folder:
                remaining = len(files) - max_files_per_folder
                items.append(f"{prefix}└── ... (还有 {remaining} 个文件)")

        except PermissionError:
            items.append(f"{prefix}❌ 无权限访问")
        except Exception as e:
            items.append(f"{prefix}❌ 错误: {str(e)}")

        return items

    print(f"📂 {os.path.basename(root_path) or root_path}")
    tree_items = _get_tree_structure(root_path)
    for item in tree_items:
        print(item)


def analyze_dataset_structure(dataset_path):
    """
    分析数据集结构并提供统计信息

    Args:
        dataset_path: 数据集根目录路径
    """
    print("=" * 60)
    print("🔍 数据集结构分析")
    print("=" * 60)

    if not os.path.exists(dataset_path):
        print(f"❌ 数据集路径不存在: {dataset_path}")
        return

    stats = {
        'total_dirs': 0,
        'total_files': 0,
        'audio_files': 0,
        'file_types': defaultdict(int),
        'dir_structure': {}
    }

    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

    for root, dirs, files in os.walk(dataset_path):
        stats['total_dirs'] += len(dirs)
        stats['total_files'] += len(files)

        # 分析文件类型
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            stats['file_types'][ext] += 1

            if ext in audio_extensions:
                stats['audio_files'] += 1

        # 分析目录结构
        rel_path = os.path.relpath(root, dataset_path)
        if rel_path == '.':
            rel_path = 'root'

        stats['dir_structure'][rel_path] = {
            'subdirs': len(dirs),
            'files': len(files),
            'audio_files': sum(1 for f in files if os.path.splitext(f)[1].lower() in audio_extensions)
        }

    # 打印统计信息
    print(f"📊 总体统计:")
    print(f"   目录总数: {stats['total_dirs']}")
    print(f"   文件总数: {stats['total_files']}")
    print(f"   音频文件: {stats['audio_files']}")

    print(f"\n📈 文件类型分布:")
    for ext, count in sorted(stats['file_types'].items()):
        if ext:
            print(f"   {ext}: {count} 个文件")
        else:
            print(f"   无扩展名: {count} 个文件")

    return stats


def show_dataset_samples(dataset_path, sample_count=3):
    """
    展示数据集中的样本文件

    Args:
        dataset_path: 数据集路径
        sample_count: 每个类别显示的样本数量
    """
    print("\n" + "=" * 60)
    print("🎯 数据集样本展示")
    print("=" * 60)

    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

    # 查找所有可能包含分类数据的目录
    class_dirs = []

    # 检查常见的数据集结构路径
    possible_paths = [
        os.path.join(dataset_path, 'train_val', 'train_data'),
        os.path.join(dataset_path, 'train_val', 'validation_data'),
        os.path.join(dataset_path, 'train_data'),
        os.path.join(dataset_path, 'validation_data'),
        dataset_path
    ]

    for path in possible_paths:
        if os.path.exists(path):
            # 检查是否包含分类子目录
            try:
                subdirs = [d for d in os.listdir(path)
                           if os.path.isdir(os.path.join(path, d))]
                if subdirs:
                    class_dirs.append((path, subdirs))
            except:
                continue

    if not class_dirs:
        print("❌ 未发现标准的分类数据结构")
        return

    for data_path, classes in class_dirs:
        data_type = os.path.basename(data_path)
        print(f"\n📁 {data_type} ({data_path}):")
        print(f"   发现 {len(classes)} 个类别")

        for class_name in classes[:10]:  # 最多显示10个类别
            class_path = os.path.join(data_path, class_name)
            try:
                files = [f for f in os.listdir(class_path)
                         if os.path.splitext(f)[1].lower() in audio_extensions]

                print(f"\n   🏷️  类别: {class_name}")
                print(f"      音频文件数: {len(files)}")

                # 显示样本文件
                sample_files = files[:sample_count]
                for i, file in enumerate(sample_files):
                    file_path = os.path.join(class_path, file)
                    file_size = os.path.getsize(file_path)
                    size_str = f"{file_size / 1024:.1f}KB" if file_size < 1024 * 1024 else f"{file_size / (1024 * 1024):.1f}MB"
                    print(f"      └── 🎵 {file} ({size_str})")

                if len(files) > sample_count:
                    print(f"      └── ... (还有 {len(files) - sample_count} 个文件)")

            except Exception as e:
                print(f"      ❌ 无法访问: {str(e)}")

        if len(classes) > 10:
            print(f"\n   └── ... (还有 {len(classes) - 10} 个类别)")


def check_mixed_data(dataset_path):
    """
    检查混合数据集结构

    Args:
        dataset_path: 数据集路径
    """
    print("\n" + "=" * 60)
    print("🔀 混合数据集检查")
    print("=" * 60)

    mix_paths = [
        os.path.join(dataset_path, 'mix_2'),
        os.path.join(dataset_path, 'mix_3')
    ]

    for mix_path in mix_paths:
        if os.path.exists(mix_path):
            mix_type = os.path.basename(mix_path)
            print(f"\n📁 {mix_type} 数据集:")

            try:
                combinations = [d for d in os.listdir(mix_path)
                                if os.path.isdir(os.path.join(mix_path, d))]

                print(f"   组合数量: {len(combinations)}")

                # 显示几个样本组合
                for combo in combinations[:5]:
                    combo_path = os.path.join(mix_path, combo)
                    try:
                        files = [f for f in os.listdir(combo_path)
                                 if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]
                        print(f"   🔀 {combo}: {len(files)} 个混合音频")
                    except:
                        print(f"   ❌ {combo}: 无法访问")

                if len(combinations) > 5:
                    print(f"   └── ... (还有 {len(combinations) - 5} 个组合)")

            except Exception as e:
                print(f"   ❌ 错误: {str(e)}")
        else:
            mix_type = os.path.basename(mix_path)
            print(f"\n❌ {mix_type} 数据集不存在")


def main():
    """主函数 - 展示完整的数据集结构"""
    # 可以修改这个路径为你的数据集路径
    dataset_path = "./audio_origin/birds"

    print("🎼 鸟类声音数据集结构分析工具")
    print("=" * 60)

    # 1. 展示目录树
    print("📂 目录树结构:")
    show_directory_tree(dataset_path, max_files_per_folder=3, max_depth=4)

    # 2. 分析数据集统计
    stats = analyze_dataset_structure(dataset_path)

    # 3. 展示数据样本
    show_dataset_samples(dataset_path, sample_count=3)

    # 4. 检查混合数据
    check_mixed_data(dataset_path)

    print("\n" + "=" * 60)
    print("✅ 数据集结构分析完成")
    print("=" * 60)


if __name__ == "__main__":
    main()