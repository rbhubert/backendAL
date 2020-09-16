from enums import sources_base


# basic random oversampling
# choose random samples of the minority class up to the number
# of documents in the majority class
def basic_oversampling(df_docs):
    only_relevant = df_docs[df_docs[sources_base.CLASSIFICATION] == "relevant"]
    only_no_relevant = df_docs[df_docs[sources_base.CLASSIFICATION] == "no_relevant"]

    number_relevants = len(only_relevant.index)
    number_no_relevants = len(only_no_relevant.index)
    difference = abs(number_no_relevants - number_relevants)

    if number_relevants > number_no_relevants:
        extras = only_no_relevant.sample(difference)
    else:
        extras = only_relevant.sample(difference)

    df_docs = df_docs.append(extras, ignore_index=True)

    return df_docs


class TypeOversampling:
    BASIC = basic_oversampling
