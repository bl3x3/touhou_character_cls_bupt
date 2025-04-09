# 2025-03-13  crawler_caller.py

# 是配合 github.com/ANewPassword/danbooru_downloader 使用的，将这个爬虫调用很多次，从而一键下载许多角色的 Gelbooru 图片。
# 脚本原名是 ls_download.py（）


import os




my_proxy_path = "127.0.0.1:7897"

character_tags: list[str] = [
    # th14
    #"wakasagihime",
    #"sekibanki",
    #"imaizumi_kagerou",
    #"tsukumo_yatsuhashi", "tsukumo_benben",
    #"kijin_seija",
    #"sukuna_shinmyoumaru",

    #"horikawa_raiko",

    # th15
    #"seiran",
    #"ringo_(touhou)",
    #"doremy_sweet",
    #"kishin_sagume",
    #"clownpiece",
    #"junko_(touhou)",

    #"hecatia_lapislazuli",

    # th16
    #"eternity_larva",
    #"sakata_nemuno",
    #"komano_aunn",
    #"yatadera_narumi",
    #"nishida_satono", "teireida_mai",
    #"matara_okina",

    # th17
    #"ebisu_eika",
    #"ushizaki_urumi",
    #"niwatari_kutaka",
    #"kicchou_yachie",
    #"joutouguu_mayumi",
    #"haniyasushin_keiki",

    #"kurokoma_saki",

    "luna_child",
    "hieda_no_akyuu",
    "reisen",
    "watatsuki_no_toyohime",
    "okunoda_miyoi",

    "star_sapphire",
    "watatsuki_no_yorihime",
    "toutetsu_yuuma",
    "yorigami_jo'on",
    "morichika_rinnosuke",

    "satsuki_rin",
]



config_json_template = '''
{{
    "args": {{
        "mode": "page",
        "template": "gelbooru",
        "start": 1,
        "end": 12,
        "tags": "{:s}+1girl+-nude+-sex+-3d",
        "path": "./download_result/{:s}",
        "proxy": "{:s}",
        "thread": 5,
        "file_config_path": "",
        "retry_max": "7",
        "log_level": "Info",
        "deduplication": "strict",
        "chksums": true,
        "with_metadata": false,
        "make_config": false,
        "no_print_log": false
    }}
}}
'''


def generate_md_file_content(character_tag: str, proxy_path: str) -> str:
    return config_json_template.format(character_tag, character_tag, proxy_path)


if __name__ == '__main__':

    for ind, character_tag in enumerate(character_tags):

        # make it more robust?
        character_tag = character_tag.strip().lower()
        print("begin downloading {:s} tag, {:d} of {:d}.".format(character_tag, ind + 1, len(character_tags)))

        config_file_content = generate_md_file_content(character_tag, my_proxy_path)
        print("generated config file content:")
        print(config_file_content)

        config_file_name = "config_{:s}.json".format(character_tag)
        with open(config_file_name, mode='w') as f:
            f.write(config_file_content)
        print("config saved to file {:s}.".format(config_file_name))

        command = "py main.py -m file \"--file-config-path\" \"./{:s}\"".format(config_file_name)
        print("now running command!")
        print("running `{}`!".format(command))
        os.system(command)




