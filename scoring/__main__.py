"""Score the flood tool based on provided sample data."""

import sys
import time

import pandas as pd
import numpy as np

import flood_tool
from .scores import SCORES

seed = 1803343

DEFAULT_DATA = flood_tool.tool.DEFAULT_UNIT_DATA.sample(frac=1.0,
                                                        random_state=seed)
RAW_FEATURES = flood_tool.tool.RAW_FEATURES

if len(sys.argv) > 1:
    pc_test_data = pd.read_csv(sys.argv[1])
    loc_test_data = pd.read_csv(sys.argv[2])
    unlabelled = pd.read_csv(sys.argv[3])
    labelled = pd.read_csv(sys.argv[4])
else:
    pc_test_data = DEFAULT_DATA.iloc[:1000, :]
    unlabelled = pc_test_data[RAW_FEATURES]
    loc_test_data = DEFAULT_DATA.iloc[1000:2000, :]
    labelled = DEFAULT_DATA.drop(pc_test_data.index)
    labelled = labelled.drop(loc_test_data.index).iloc[:8000, :]

tool = flood_tool.Tool(unlabelled,
                       labelled)


print("Scoring flood class from postcode models")
print("========================================\n")
models = flood_tool.flood_class_from_postcode_models

for model, name in list(models.items())[:3]:
    t1 = time.time()
    try:
        tool.fit_to_data(models=[model])
    except Exception as e:

        print(f"{name}: training failed with {e}.")
        continue
    t2 = time.time()
    print(f"{name}: training time {t2-t1:0.5f} s")
    t1 = time.time()
    try:
        prediction = tool.predict_flood_class_from_postcode(
            pc_test_data.postcode, model
        )
    except Exception as e:
        print(f"{name}: prediction failed with {e}.")
        continue
    t2 = time.time()
    print(f"{name}: prediction time {t2-t1:0.5f} s")
    prediction.reindex(pc_test_data.postcode)
    score = sum(
        [
            SCORES[_p - 1, _t - 1]
            for _p, _t in zip(prediction, pc_test_data.riskLabel)
        ]
    )
    print(f"{name}: score {score}")


print("\nScoring flood class from location models")
print("========================================\n")
models = flood_tool.flood_class_from_location_models

for model, name in list(models.items())[:3]:
    t1 = time.time()
    try:
        tool.fit_to_data(models=[model])
    except Exception as e:
        print(f"{name}: training failed with {e}.")
        continue
    t2 = time.time()
    print(f"{name}: training time {t2-t1:0.5f} s")
    t1 = time.time()
    try:
        prediction = tool.predict_flood_class_from_OSGB36_location(
            loc_test_data.easting, loc_test_data.northing, model
        )
    except Exception as e:
        print(f"{name}: prediction failed with {e}.")
        continue
    t2 = time.time()
    print(f"{name}: prediction time {t2-t1:0.5f} s")
    prediction.reindex([(east, north) for
                        east, north in zip(loc_test_data.easting,
                                           loc_test_data.northing)])
    score = sum(
        [
            SCORES[_p - 1, _t - 1]
            for _p, _t in zip(prediction, loc_test_data.riskLabel)
        ]
    )
    print(f"{name}: score {score}")


print("\nScoring historic flooding models")
print("=================================\n")
models = flood_tool.historic_flooding_models

for model, name in list(models.items())[:3]:
    t1 = time.time()
    try:
        tool.fit_to_data(models=[model])
    except Exception as e:
        print(f"{name}: training failed with {e}.")
        continue
    t2 = time.time()
    print(f"{name}: training time {t2-t1:0.5f} s")
    t1 = time.time()
    try:
        prediction = tool.predict_historic_flooding(
            pc_test_data.postcode.to_list(), model
        )
    except Exception as e:
        print(f"{name}: prediction failed with {e}.")
        continue
    t2 = time.time()
    print(f"{name}: prediction time {t2-t1:0.5f} s")

    prediction = prediction.reindex(pc_test_data['postcode'])

    truth = pc_test_data.set_index("postcode")
    truth = truth['historicallyFlooded']

    tps = sum(prediction & truth)
    fps = sum(prediction & ~truth)
    tns = sum(~prediction & ~truth)
    fns = sum(~prediction & truth)

    if tps+fps == 0:
        print(f"{name}: precision nan")
    else:
        print(f"{name}: precision {tps/(tps+fps):0.3f}")
    if tps+fns == 0:
        print(f"{name}: recall nan")
    else:
        print(f"{name}: recall {tps/(tps+fns):0.3f}")
    print(f"{name}: accuracy {(tps+tns)/(tps+tns+fps+fns):0.3f}")
    if 2*tps+fps+fns == 0:
        print(f"{name}: f1 score nan")
    else:
        print(f"{name}: f1 score {2*tps/(2*tps+fps+fns):0.3f}")


print("\nScoring house price models")
print("===========================\n")
models = flood_tool.house_price_models

for model, name in list(models.items())[:3]:
    t1 = time.time()
    try:
        tool.fit_to_data(models=[model])
    except Exception as e:
        print(f"{name}: training failed with {e}.")
        continue
    t2 = time.time()
    print(f"{name}: training time {t2-t1:0.5f} s")
    t1 = time.time()
    try:
        prediction = tool.predict_median_house_price(
            pc_test_data.postcode.to_list(), model
        )
    except Exception as e:
        print(f"{name}: prediction failed with {e}.")
        continue
    t2 = time.time()
    print(f"{name}: prediction time {t2-t1:0.5f} s")

    prediction = prediction.reindex(pc_test_data.postcode)

    truth = pc_test_data.set_index("postcode")
    truth = truth['medianPrice']

    valid = (prediction.notna() & truth.notna())

    score = np.sqrt(np.mean((prediction.loc[valid]
                             - truth.loc[valid])**2))

    fps = sum(prediction.isna() & truth.notna())
    fns = sum(prediction.isna() & truth.isna())

    print(f"{name}: score {score:0.2f}")
    print(f"{name}: false positive NaNs {fps}")
    print(f"{name}: false negative NaNs {fns}")


print("\nScoring local authority models")
print("===============================\n")
models = flood_tool.local_authority_models

for model, name in list(models.items())[:3]:
    t1 = time.time()
    try:
        tool.fit_to_data(models=[model])
    except Exception as e:
        print(f"{name}: training failed with {e}.")
        continue
    t2 = time.time()
    print(f"{name}: training time {t2-t1:0.5f} s")
    t1 = time.time()
    try:
        prediction = tool.predict_local_authority(
            loc_test_data.easting.to_list(),
            loc_test_data.northing.to_list(),
            model
        )
    except Exception as e:
        print(f"{name}: prediction failed with {e}.")
        continue
    t2 = time.time()
    print(f"{name}: prediction time {t2-t1:0.5f} s")

    locations = [(east, north) for
                 east, north in zip(loc_test_data.easting,
                                    loc_test_data.northing)]

    truth = loc_test_data.set_index(pd.Series(locations,
                                              index=loc_test_data.index))
    truth = truth.localAuthority

    tps = sum(prediction == truth)

    print(f"{name}: accuracy {tps/len(truth):0.3f}")
