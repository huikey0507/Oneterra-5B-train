cd /mnt_shui

# 完整复制项目，但排除 3 个超大文件夹
rsync -av \
--exclude=checkpoints \
--exclude=datas \
--exclude=inits \
--exclude=wkdrs \
/mnt_llm_A100_V1/shui/LAE/OneTerra-train/ ./OneTerra-train/

# 进入新目录，把 3 个大文件夹软链接过来
cd OneTerra-train
ln -s /mnt_llm_A100_V1/shui/LAE/OneTerra-train/checkpoints ./
ln -s /mnt_llm_A100_V1/shui/LAE/OneTerra-train/datas ./
ln -s /mnt_llm_A100_V1/shui/LAE/OneTerra-train/inits ./
In -s /mnt_llm_A100_V1/shui/LAE/OneTerra-train/wkdrs ./