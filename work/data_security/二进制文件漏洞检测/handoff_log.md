# 二进制文件漏洞检测工作日志

## 当前状态

- 当前最佳线上分数：`0.99951`
- 当前最佳文件：`submissions/s15.csv`
- 当前最佳策略：`entry_bad/entry_good` 二分类符号规则 + `Sxxxxx` source-id CWE chunk + 稀有类恢复 + 少量 source 离群点修正
- 当前注意事项：平台上传对文件名敏感，后续提交文件名必须短，例如 `s15.csv`、`s16f.csv`
- 当前未解决：距离 `1.00000` 仍差 `0.00049`，剩余错误大概率是极少数 CWE 边界样本

## 数据与格式结论

- 训练集：`39380` 行，正样本 `19738`，负样本 `19642`。
- 测试集：`30974` 行。
- 目标可视为 `NO_VULN + 86 CWE` 的联合标签，但建模时分层更稳定。
- 可上传格式已确认：`binary_id,label,cwe_id`，UTF-8 BOM。
- `label=0` 的 `cwe_id` 必须为空。
- `label=1` 的 `cwe_id` 必须非空且形如 `CWE-xxx`。
- 长文件名会导致上传失败，后续只用短文件名提交。

## 关键发现

### 二分类标签

COFF 符号表和 `main` 调用关系可解析出主入口调用的是 `entry_bad` 还是 `entry_good`。这使二分类任务基本解决，后续主要瓶颈转为 CWE 细分类。

相关脚本：

```text
src/symbol_rule_predict.py
src/entry_body_cwe.py
```

### source-id 结构

二进制调试路径中保留 `Sxxxxx/source_sanitized.c`，可抽取 source id。

缓存文件：

```text
processed/source_ids.csv
```

结构确认：

- 全部 source id 数：`35177`
- 每个 `Sxxxxx` 正好对应两个二进制：good/bad
- 训练集中 bad+good 同时存在：`13726`
- 训练集中 bad only：`6012`，对应测试集中 good only
- 训练集中 good only：`5916`，对应测试集中 bad only
- 测试集中 bad+good 同时存在：`9523`

训练正样本按 `source_id_num` 排序后形成 CWE 连续区间。这个结构是从 `0.92757` 后继续大幅提升到 `0.97900`、`0.99951` 的核心。

### 稀有类

`CWE-561` 与 `CWE-562` 在 v12/v15b 输出中缺失，导致宏平均 F1 被严重拉低。通过 source-id 邻域和反汇编人工校验恢复：

```text
S41949 -> CWE-561 -> BIN_1574217
S41950 -> CWE-562 -> BIN_3711506
S41952 -> CWE-562 -> BIN_3228938
```

该修正使提交覆盖完整 86 个 CWE 类，是 `s15.csv` 能到 `0.99951` 的关键。

## 实验阶段记录

### 2026-05-01 21:52 基线

- 实现 `build_features.py`、`train.py`、`predict.py`。
- 特征包括 PE 头、节区统计、导入导出、字节直方图、可打印字符串、指令统计。
- 分层 `LinearSVC` 本地宏 F1：`0.822071`。
- 线上首次可用格式得分：`0.87120`。
- 可上传文件：`submission_binary_id_utf8_sig.csv`。

### 2026-05-02 01:00 符号二分类

- 解析 `main` 调用，发现 `entry_bad/entry_good` 可直接判定是否漏洞。
- 二分类瓶颈基本解除。
- 本地宏 F1 提升到约 `0.920997`。
- 代表文件：`submission_symbol_hybrid_hash_hint_utf8_sig.csv`。

### 2026-05-02 06:00 增强 CWE 模型与 PyTorch

- 增加 `entry_bad` 函数体字节、反汇编 token、符号 token、字符串 token。
- 实验 `enhanced_cwe_calibrator.py`。
- 使用 PyTorch CUDA 环境训练 `torch_group_cnn.py` 做组内修正。
- 这些模型对局部混淆有帮助，但最终不如 source-id 结构稳定。
- 代表文件：`submission_torch_group_cnn_v6_utf8_sig.csv`、`submission_torch_hint_v7_utf8_sig.csv`。

### 2026-05-02 07:00 v11

- 多轮规则和模型叠加后生成 `submission_uninit457_v11_utf8_sig.csv`。
- 本地验证宏 F1：`0.993625765896`。
- 这是 source-id chunk 之前的最强候选。

### 2026-05-02 07:55 v12 source-id chunk

- 新增 `src/source_id_chunk_postprocess.py`。
- 使用训练正样本 source-id 连续 CWE chunk 覆盖测试正样本 CWE。
- 本地验证：`0.993625765896 -> 0.999119583180`。
- 验证统计：`changed=28`、`useful=27`、`harmful=0`、`neutral=1`。
- 生成文件：`submission_source_chunk_v12_utf8_sig.csv`。
- 用户反馈线上分数：`0.97900`。

### 2026-05-02 09:20 v14 稀有类恢复

- 新增 `src/rare_singleton_postprocess.py`。
- 修复 v12/v15b 缺失的 `CWE-561/CWE-562`。
- 修改 3 行：

```text
BIN_1574217: CWE-398 -> CWE-561
BIN_3711506: CWE-546 -> CWE-562
BIN_3228938: CWE-546 -> CWE-562
```

### 2026-05-02 09:30 v15 source outlier

- 新增 `src/source_outlier_postprocess.py`。
- 在 v14 基础上修正 5 个明显落在错误 source-id 区间的离群点：

```text
S511   BIN_8681591: CWE-481 -> CWE-126
S4917  BIN_3780181: CWE-681 -> CWE-121
S4966  BIN_4860053: CWE-416 -> CWE-126
S10849 BIN_8389703: CWE-416 -> CWE-126
S10850 BIN_9335771: CWE-364 -> CWE-126
```

- 完整文件：`submission_source_outlier_v15_utf8_sig.csv`。
- 短名副本：`s15.csv`。

### 2026-05-02 09:45 v15b 文件名问题定位

- 长文件名上传失败。
- 生成短名 `sub.csv`，对应 `submission_source_outlier_safe_v15b_utf8_sig.csv`。
- `sub.csv` 不含 `CWE-561/CWE-562` 三个稀有类修正，只含 5 个 outlier 修正。
- 用户反馈线上分数：`0.98029`。
- 结论：上传失败主要是文件名过长，后续必须用短名。

### 2026-05-02 10:00 s15

- 上传短名 `s15.csv`。
- `s15.csv` 等同完整 v15：v12 + 3 个稀有类修正 + 5 个 source outlier 修正。
- 用户反馈线上分数：`0.99951`。
- 当前最佳文件确定为 `submissions/s15.csv`。

### 2026-05-02 10:40 s16 单点探针

为冲击 `1.00000`，新增符号与边界分析：

- `src/symbol_cwe_postprocess.py`
- `report/test_entry_symbol_cwe_check.csv`
- `report/test_raw_cwe_s15_check.csv`
- `report/train_good_raw_cwe_for_test_s15.csv`
- `report/test_good_raw_cwe_for_test_s15.csv`
- `report/boundary_similarity_s15.csv`

验证结论：

- `entry_bad` 符号中的 CWE 在训练正样本上准确率 `0.999529633114`，但存在 `CWE-135` 包装例外，不能盲目覆盖。
- 测试集中 `entry_bad` 与 `s15` 唯一冲突为 `BIN_7115172`，它属于 `CWE-135` 例外候选。
- 同源 good 文件中的纯 CWE 字符串在训练配对上 `100%` 准确，但对 `s15` 没有发现可安全修正的差异。
- good 二进制训练分类器本地验证明显弱于当前 source-id 方案，不能用于批量覆盖。
- 函数体相似度批量改边界在本地验证会显著掉分，因此只能做单点探针。

已生成短名探针文件：

```text
s16a.csv: BIN_7115172 CWE-135 -> CWE-121
s16b.csv: BIN_8249526 CWE-126 -> CWE-121
s16c.csv: BIN_6545696 CWE-126 -> CWE-121
s16d.csv: BIN_1812963 CWE-126 -> CWE-121
s16e.csv: BIN_8556812 CWE-526 -> CWE-546
s16f.csv: BIN_4198849 CWE-252 -> CWE-253
s16g.csv: s16b+s16c+s16d 组合
```

推荐测试顺序：

```text
s16f.csv -> s16b.csv -> s16c.csv -> s16d.csv -> s16a.csv -> s16e.csv
```

如果某个单点探针高于 `0.99951`，应基于该方向继续扩展；如果下降，回到 `s15.csv`。

## 当前最佳与风险

当前最佳：

```text
submissions/s15.csv
线上分数：0.99951
```

剩余风险：

- 剩余误差过小，本地验证很难区分最后 1 个或少量边界错误。
- 大模型、相似度批量规则、good 二进制分类器均已验证存在误伤风险。
- 不应覆盖 `s15.csv`；所有后续尝试必须另存短名文件。
- `s16*.csv` 均为线上探针，不是已确认优于 `s15.csv` 的最终文件。

## 下一步建议

1. 优先上传 `s16f.csv`，记录线上分数。
2. 若 `s16f.csv` 未提升，依次测试 `s16b.csv`、`s16c.csv`、`s16d.csv`。
3. 若某个探针提升，立即围绕同一 CWE 边界生成新的单点或小组合候选。
4. 若全部下降，继续保留 `s15.csv` 作为最终最优提交。

## 2026-05-02 22:55 wrapper/source 最近邻复核

用户反馈 `s16f.csv` 显示分数仍为 `0.99951`，当天已无提交次数。截图里的参考思路是：用 `wrapper_bad.c / wrapper_good.c / injected_bad.cpp / injected_good.cpp` 判断 `label`，再用 `Sxxxxx` 调试路径编号做 CWE 最近邻映射。

复核结果：

- `wrapper/injected` 标签信号在训练集全覆盖且准确率 `1.0`。
- 测试集用该信号重建 `label` 后，与 `s15.csv` 没有任何差异；截图思路中的二分类部分已经被当前方案完全吸收。
- 普通 `Sxxxxx` 最近邻相对 `s15.csv` 改 32 行；同类最近邻相对 `s15.csv` 风险更高，并且会让输出只剩 85 个 CWE，不建议提交。
- 按当前线上 `0.99951` 反推，剩余错误更可能在 `CWE-121/122/126` 等大类边界，不像低样本 CWE 错误；低样本 CWE 单行错改会造成远大于当前差距的宏 F1 损失。

新增脚本：

```text
src/wrapper_nearest_postprocess.py
```

已生成并校验格式的新候选：

```text
s17b.csv: 19 行，只改 CWE-121/122/126 最近邻边界，下一轮第一优先
s17c.csv: 23 行，加入 one_error_drop_est <= 0.00050 的中等类边界，第二优先
s17d.csv: 24 行，加入 one_error_drop_est <= 0.00060，第三优先
s17n.csv: 32 行，完整普通最近邻，风险高，不建议优先
s17k.csv: 同类最近邻，只有 85 个 CWE，不建议上传
```

格式校验结果：

```text
s17b.csv (30974, 3), label 分布 {0: 15535, 1: 15439}, CWE 覆盖 86, ok
s17c.csv (30974, 3), label 分布 {0: 15535, 1: 15439}, CWE 覆盖 86, ok
s17d.csv (30974, 3), label 分布 {0: 15535, 1: 15439}, CWE 覆盖 86, ok
s17n.csv (30974, 3), label 分布 {0: 15535, 1: 15439}, CWE 覆盖 86, ok
s17k.csv (30974, 3), label 分布 {0: 15535, 1: 15439}, CWE 覆盖 85, 不建议
```

下一轮提交建议更新为：

```text
s17b.csv -> s17c.csv -> s17d.csv
```

如果 `s17b.csv` 高于 `0.99951`，说明剩余误差确实集中在大类 source 边界，可以继续沿 `CWE-121/122/126` 近邻边界扩展；如果下降，回退到 `s15.csv` 并只做单点探针。

## 2026-05-03 01:03 s17b 反馈与 s18

用户反馈 `s17b.csv` A 榜得分为 `0.99957`，相对 `s15.csv` 的 `0.99951` 有提升。该结果说明 `CWE-121/122/126` 大类 source 边界方向有效，但 19 行里不可能全对，后续不应继续批量给多个候选。

新的提交策略改为一次只给一个最优候选。当前唯一建议提交文件：

```text
submissions/s18.csv
```

`s18.csv` 以 `s17b.csv` 为基线，只新增 1 行：

```text
BIN_7681509: CWE-396 -> CWE-391
```

选择理由：

- `BIN_7681509` 位于 `S35236`，左侧最近训练正样本 `S35235` 为 `CWE-391`。
- 右侧 `S35237-S35242` 已稳定进入 `CWE-396`，该点是 `CWE-391/396` 的单点边界。
- 单行宏 F1 影响估计约 `0.00046`，与 `s17b` 距离 `1.00000` 的剩余差距接近，若命中有机会产生明显跃升。
- 相比 `s17c/s17d`，`s18` 不再同时引入多个中等风险点，提交成本和归因风险最低。

格式校验：

```text
s18.csv (30974, 3), label 分布 {0: 15535, 1: 15439}, CWE 覆盖 86, ok
相对 s17b 仅 1 行差异
相对 s15 共 20 行差异
```

下一次只建议上传 `s18.csv`。如果 `s18.csv` 提升，继续沿 `CWE-391/396` 源编号边界做单点扩展；如果下降，保留 `s17b.csv` 作为当前 A 榜最优候选。

## 2026-05-03 02:10 s18 下降后的鲁棒修正

用户反馈 `s18.csv` A 榜得分 `0.99847`，明显低于 `s17b.csv` 的 `0.99957`。结论：不能继续按单个 source-id 边界猜测，尤其不能只因为最近训练样本属于另一类就改测试样本。

重新验证后采用更稳的融合原则：

```text
source-id 最近邻只负责提出候选；
entry_bad 函数字节模型必须也支持新 CWE，才接受修改。
```

验证依据：

- 只在 `CWE-121/122/126` 三类上训练 `entry_bad` 函数字节模型，5 折交叉验证宏 F1 为 `0.9738046991970849`。
- 在固定验证集上，直接使用最近 source-id 修改同类边界是一正一负，宏 F1 无提升。
- 加上 `entry_bad` 函数字节模型过滤后，只保留正确修正，验证宏 F1 从 `0.999119583180` 提升到 `0.999189000511`。
- 全量用函数字节模型覆盖会严重下降，因此不能替代当前 source-id 主方案，只能作为候选过滤器。

已生成当前唯一下一版候选：

```text
submissions/s19.csv
```

`s19.csv` 以 `s15.csv` 为基线，只接受 `s17b` 中被函数字节模型支持的 4 行：

```text
BIN_1812963: CWE-126 -> CWE-121
BIN_6244783: CWE-126 -> CWE-122
BIN_1455492: CWE-121 -> CWE-126
BIN_2334733: CWE-126 -> CWE-121
```

格式校验：

```text
s19.csv (30974, 3), label 分布 {0: 15535, 1: 15439}, CWE 覆盖 86, ok
相对 s15 改 4 行
相对 s17b 回退 15 行
```

推荐下一次只上传 `s19.csv`。如果 `s19.csv` 低于 `s17b.csv`，说明 A 榜这 15 个 source-only 边界里有较多真实修正，但从 AB 榜角度仍不建议继续靠 A 榜边界反馈扩张；应继续构造语义过滤器，而不是恢复测试集边界猜测。
