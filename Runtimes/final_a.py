# Auto-generated from final_a.ipynb code cells

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('TIMING.csv')


rot_fit_time = float(df[df['ID'] == 'rot_train___128_1e-05_0.5'].iloc[0]['time'])
rot_explain_time = float(df[df['ID'] == 'rot_explanation'].iloc[0]['time'])
rot_combined = float(df[df['ID'] == 'rot_combined'].iloc[0]['time'])


df_embedding = df[df['ID'] == 'embedding_generation']
df_shap = df[df['ID'] == 'shap']
df_ig = df[df['ID'] == 'ig']
df_lime = df[df['ID'] == 'lime_5000']
# df_shap=df_shap.sort_values(by='time')


df_shap['cumm_time'] = df_shap['time'].cumsum()
df_shap['avg_time'] = 1
df_shap['avg_time'] = df_shap['cumm_time'] / df_shap['avg_time'].cumsum()
# df_shap.head()


df_lime['cumm_time'] = df_lime['time'].cumsum()
df_lime['avg_time'] = 1
df_lime['avg_time'] = df_lime['cumm_time'] / df_lime['avg_time'].cumsum()


df_shap['rot_cumm'] = rot_combined
df_shap['rot_cumm'] += np.linspace(0,rot_explain_time, len(df_shap)+1)[1:]
df_shap['rot_avg'] = 1
df_shap['rot_avg'] = df_shap['rot_avg'].cumsum()
df_shap['rot_avg'] = df_shap['rot_cumm'] / df_shap['rot_avg']


df_shap['case_number'] = [i+1 for i in range(len(df_shap))]
df_ig['case_number'] = [i+1 for i in range(len(df_ig))]
df_lime['case_number'] = [i+1 for i in range(len(df_lime))]


# df_shap


plt.plot(df_shap['case_number'], df_shap['rot_cumm'], label='RoT total time', marker=',')
plt.plot(df_shap['case_number'], df_shap['cumm_time'], label='SHAP total time', marker='.')
plt.plot(df_lime['case_number'], df_lime['cumm_time'], label='LIME total time', marker='^')
plt.xlabel('num cases')
plt.ylabel('time taken (seconds, total)')
plt.yscale('log')
plt.legend()



plt.plot(df_shap['case_number'], df_shap['rot_avg'], label='RoT average time', marker=',')
plt.plot(df_shap['case_number'], df_shap['avg_time'], label='SHAP average time', marker='.')
plt.plot(df_lime['case_number'], df_lime['avg_time'], label='LIME average time', marker='^')
# plt.axhline(np.mean(df_shap['avg_time']), label='SHAP overall average', linestyle=':')
# plt.axhline(df_shap['rot_cumm'].values.tolist()[-1], label='RoT total', linestyle=':')
plt.xlabel('num cases')
plt.ylabel('time taken (seconds, average)')
plt.legend()



# df_shap.describe()


rot_times = [rot_explain_time / len(df_shap)] * len(df_shap)
rot_times[0] += rot_combined


plt.figure(figsize=(6,6))
plt.plot(df_shap['case_number'], rot_times, label='RoT', marker='.', linewidth=1, ms=12, zorder=3)
plt.plot(df_shap['case_number'], df_shap['time'], label='SHAP', marker='x', linewidth=1, ms=7, zorder=1)
plt.plot(df_ig['case_number'], df_ig['time'], label='Int. Grad.', marker='2', linewidth=1, ms=9, zorder=2, color='C2')
plt.plot(df_lime['case_number'], df_lime['time'], label='LIME', marker='^', linewidth=1, ms=8, zorder=2, color='C3')
plt.axhline(np.mean(rot_times), label='RoT average', linestyle=':', linewidth=3, zorder=1)
plt.axhline(np.mean(df_shap['time']), label='SHAP average', linestyle=':', color='tab:orange', linewidth=3, zorder=2)
plt.axhline(np.mean(df_ig['time']), label='I.G. average', linestyle=':', color='C2', linewidth=3, zorder=2)
# plt.axhline(df_shap['rot_cumm'].values.tolist()[-1], label='RoT total', linestyle=':')
plt.xlabel('Case Number', fontsize=24)
plt.ylabel('Time Taken (log)', fontsize=24)
# plt.title('RoT vs SHAP: Time to Compute Explanations')
# plt.yscale('symlog')
# plt.ylim(0,8000)
plt.yticks(fontsize=19)
plt.xticks(fontsize=19)
plt.yscale('log')
plt.legend(fontsize=19, borderpad=0.1, borderaxespad=0.2, loc="lower right", bbox_to_anchor=(1,0.07))
plt.tight_layout()
plt.savefig('unsorted_runtimes.pdf', dpi=300)





